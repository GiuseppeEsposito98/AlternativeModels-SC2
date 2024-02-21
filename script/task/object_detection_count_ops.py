import argparse
import builtins as __builtin__
import datetime
import json
import os
import time

import torch
from torch import distributed as dist
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data._utils.collate import default_collate
from torchdistill.common import file_util, module_util, yaml_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import is_main_process, init_distributed_mode, load_ckpt, save_ckpt, set_seed
from torchdistill.core.distillation import get_distillation_box
from torchdistill.core.training import get_training_box
from torchdistill.datasets import util
from torchdistill.datasets.coco import get_coco_api_from_dataset
from torchdistill.eval.coco import CocoEvaluator
from torchdistill.misc.log import setup_log_file, SmoothedValue, MetricLogger
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection import ssd300_vgg16

from sc2bench.analysis import check_if_analyzable
from sc2bench.common.config_util import overwrite_config
from sc2bench.models.detection.base import check_if_updatable_detection_model
from sc2bench.models.detection.registry import load_detection_model
from sc2bench.models.detection.wrapper import get_wrapped_detection_model

logger = def_logger.getChild(__name__)
torch.multiprocessing.set_sharing_strategy('file_system')


def get_argparser():
    parser = argparse.ArgumentParser(description='Supervised compression for object detection tasks')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--json', help='json string to overwrite config')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--log', help='log file path')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--iou_types', nargs='+', help='IoU types for evaluation '
                                                       '(the first IoU type is used for checkpoint selection)')
    parser.add_argument('--seed', type=int, help='seed in random number generator')
    parser.add_argument('-test_only', action='store_true', help='only test the models')
    parser.add_argument('-student_only', action='store_true', help='test the student model only')
    parser.add_argument('-no_dp_eval', action='store_true',
                        help='perform evaluation without DistributedDataParallel/DataParallel')
    parser.add_argument('-log_config', action='store_true', help='log config')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('-adjust_lr', action='store_true',
                        help='multiply learning rate by number of distributed processes (world_size)')
    return parser

def count_conv2d(m, x, y):
    x = x[0]

    cin = m.in_channels // m.groups
    cout = m.out_channels // m.groups
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

    # ops per output element
    kernel_mul = kh * kw * cin
    kernel_add = kh * kw * cin - 1
    bias_ops = 1 if m.bias is not None else 0
    ops = kernel_mul + kernel_add + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops

    m.total_adds += torch.Tensor([int((kernel_add + bias_ops)*num_out_elements)])
    m.total_mults += torch.Tensor([int(kernel_mul*num_out_elements)])

    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])

def count_bn2d(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_sub = nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_subs += torch.Tensor([int(nelements)])
    m.total_divs += torch.Tensor([int(nelements)])
    m.total_ops += torch.Tensor([int(total_ops)])

def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_comps += torch.Tensor([int(nelements)]) *2
    m.total_ops += torch.Tensor([int(total_ops)])

def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_divs += torch.Tensor([int(total_div)])
    m.total_exps += torch.Tensor([int(total_exp)])
    m.total_adds += torch.Tensor([int(total_add)])
    m.total_ops += torch.Tensor([int(total_ops)])

def count_maxpool(m, x, y):
    kernel_ops = torch.prod(torch.Tensor([m.kernel_size])) - 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_comps += torch.Tensor([int(kernel_ops*num_elements)])
    m.total_ops += torch.Tensor([int(total_ops)])

def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_divs += torch.Tensor([int(total_div*num_elements)])
    m.total_adds += torch.Tensor([int(total_add*num_elements)])
    m.total_ops += torch.Tensor([int(total_ops)])

def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements

    m.total_mults += torch.Tensor([int(total_mul*num_elements)])
    m.total_adds += torch.Tensor([int(total_add*num_elements)])
    m.total_ops += torch.Tensor([int(total_ops)])

def profile(model, input_size, custom_ops = {}):

    model.eval().cpu()
    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_comps', torch.zeros(1))
        m.register_buffer('total_divs', torch.zeros(1))
        m.register_buffer('total_mults', torch.zeros(1))
        m.register_buffer('total_adds', torch.zeros(1))
        m.register_buffer('total_exps', torch.zeros(1))
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_subs', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, torch.nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d)):
            m.register_forward_hook(count_maxpool)
        elif isinstance(m, (torch.nn.AvgPool1d, torch.nn.AvgPool2d, torch.nn.AvgPool3d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, torch.nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, torch.nn.Hardswish):
            m.register_forward_hook(count_relu)
        elif isinstance(m, torch.nn.Hardtanh):
            m.register_forward_hook(count_relu)
        elif isinstance(m, torch.nn.Hardsigmoid):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            pass
        else:
            print("Not implemented for ", m)

    model.apply(add_hooks)

    x = torch.zeros(input_size)
    model(x)

    total_ops = 0

    total_params = 0
    total_comps = 0
    total_divs = 0
    total_mults = 0
    total_adds = 0
    total_exps = 0
    total_subs = 0
    for m in model.modules():
        if len(list(m.children())) > 0: continue
        total_ops += m.total_ops
        total_comps += m.total_comps
        total_divs += m.total_divs
        total_mults += m.total_mults
        total_adds += m.total_adds
        total_exps += m.total_exps
        total_subs += m.total_subs
        total_params += m.total_params
    
    total_comps = total_comps
    total_divs = total_divs
    total_mults = total_mults
    total_adds = total_adds
    total_exps = total_exps
    total_subs = total_subs
    total_params = total_params
    total_ops = total_ops
    total_params = total_params

    return total_adds, total_subs, total_mults, total_divs, total_exps, total_comps, total_ops, total_params


def load_model(model_config, device):
    if 'detection_model' not in model_config:
        return load_detection_model(model_config, device)
    return get_wrapped_detection_model(model_config, device)


def train_one_epoch(training_box, aux_module, bottleneck_updated, device, epoch, log_freq):
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value}'))
    uses_aux_loss = aux_module is not None and not bottleneck_updated
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch, targets, supp_dict in \
            metric_logger.log_every(training_box.train_data_loader, log_freq, header):
        sample_batch = list(image.to(device) for image in sample_batch)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        start_time = time.time()
        supp_dict = default_collate(supp_dict)
        loss = training_box(sample_batch, targets, supp_dict)
        aux_loss = None
        if uses_aux_loss:
            aux_loss = aux_module.aux_loss()
            aux_loss.backward()

        training_box.update_params(loss)
        batch_size = len(sample_batch)
        if uses_aux_loss:
            metric_logger.update(loss=loss.item(), aux_loss=aux_loss.item(),
                                 lr=training_box.optimizer.param_groups[0]['lr'])
        else:
            metric_logger.update(loss=loss.item(), lr=training_box.optimizer.param_groups[0]['lr'])

        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))
        if (torch.isnan(loss) or torch.isinf(loss)) and is_main_process():
            raise ValueError('The training loop was broken due to loss = {}'.format(loss))


def get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, DistributedDataParallel):
        model_without_ddp = model.module

    iou_type_list = ['bbox']
    if isinstance(model_without_ddp, MaskRCNN):
        iou_type_list.append('segm')
    if isinstance(model_without_ddp, KeypointRCNN):
        iou_type_list.append('keypoints')
    return iou_type_list


def log_info(*args, **kwargs):
    force = kwargs.pop('force', False)
    if is_main_process() or force:
        logger.info(*args, **kwargs)


@torch.inference_mode()
def evaluate(model_wo_ddp, data_loader, iou_types, device, device_ids, distributed, no_dp_eval=False,
             log_freq=1000, title=None, header='Test:'):
    model = model_wo_ddp.to(device)
    if distributed and not no_dp_eval:
        model = DistributedDataParallel(model, device_ids=device_ids)
    elif device.type.startswith('cuda') and not no_dp_eval:
        model = DataParallel(model, device_ids=device_ids)
    elif hasattr(model, 'use_cpu4compression'):
        model.use_cpu4compression()

    if title is not None:
        logger.info(title)

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)

    # Replace built-in print function with logger.info to log summary printed by pycocotools
    builtin_print = __builtin__.print
    __builtin__.print = log_info

    cpu_device = torch.device('cpu')
    model.eval()
    analyzable = check_if_analyzable(model_wo_ddp)
    metric_logger = MetricLogger(delimiter='  ')
    coco = get_coco_api_from_dataset(data_loader.dataset)
    if iou_types is None or (isinstance(iou_types, (list, tuple)) and len(iou_types) == 0):
        iou_types = get_iou_types(model)

    coco_evaluator = CocoEvaluator(coco, iou_types)
    for sample_batch, targets in metric_logger.log_every(data_loader, log_freq, header):
        sample_batch = list(image.to(device) for image in sample_batch)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(sample_batch)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    avg_stats_str = 'Averaged stats: {}'.format(metric_logger)
    logger.info(avg_stats_str)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # Revert print function
    __builtin__.print = builtin_print

    torch.set_num_threads(n_threads)
    if analyzable and model_wo_ddp.activated_analysis:
        model_wo_ddp.summarize()
    return coco_evaluator


def train(teacher_model, student_model, dataset_dict, ckpt_file_path, device, device_ids, distributed, config, args):
    logger.info('Start training')
    train_config = config['train']
    lr_factor = args.world_size if distributed and args.adjust_lr else 1
    training_box = get_training_box(student_model, dataset_dict, train_config,
                                    device, device_ids, distributed, lr_factor) if teacher_model is None \
        else get_distillation_box(teacher_model, student_model, dataset_dict, train_config,
                                  device, device_ids, distributed, lr_factor)
    best_val_map = 0.0
    optimizer, lr_scheduler = training_box.optimizer, training_box.lr_scheduler
    if file_util.check_if_exists(ckpt_file_path):
        best_val_map, _, _ = load_ckpt(ckpt_file_path, optimizer=optimizer, lr_scheduler=lr_scheduler)

    log_freq = train_config['log_freq']
    iou_types = args.iou_types
    val_iou_type = iou_types[0] if isinstance(iou_types, (list, tuple)) and len(iou_types) > 0 else 'bbox'
    student_model_without_ddp = student_model.module if module_util.check_if_wrapped(student_model) else student_model
    # aux_module = student_model_without_ddp.get_aux_module() \
    #     if check_if_updatable_detection_model(student_model_without_ddp) else None
    aux_module = None
    epoch_to_update = train_config.get('epoch_to_update', None)
    bottleneck_updated = False
    no_dp_eval = args.no_dp_eval
    start_time = time.time()
    for epoch in range(args.start_epoch, training_box.num_epochs):
        training_box.pre_process(epoch=epoch)
        if epoch_to_update is not None and epoch_to_update <= epoch and not bottleneck_updated:
            logger.info('Updating entropy bottleneck')
            student_model_without_ddp.update()
            bottleneck_updated = True

        train_one_epoch(training_box, aux_module, bottleneck_updated, device, epoch, log_freq)
        val_coco_evaluator =\
            evaluate(student_model, training_box.val_data_loader, iou_types, device, device_ids, distributed,
                     no_dp_eval=no_dp_eval, log_freq=log_freq, header='Validation:')
        # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
        val_map = val_coco_evaluator.coco_eval[val_iou_type].stats[0]
        if val_map > best_val_map and is_main_process():
            logger.info('Best mAP ({}): {:.4f} -> {:.4f}'.format(val_iou_type, best_val_map, val_map))
            logger.info('Updating ckpt at {}'.format(ckpt_file_path))
            best_val_map = val_map
            save_ckpt(student_model_without_ddp, optimizer, lr_scheduler,
                      best_val_map, config, args, ckpt_file_path)
        training_box.post_process()

    if distributed:
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    training_box.clean_modules()


def main(args):
    log_file_path = args.log
    if is_main_process() and log_file_path is not None:
        setup_log_file(os.path.expanduser(log_file_path))

    distributed, device_ids = init_distributed_mode(args.world_size, args.dist_url)
    logger.info(args)
    cudnn.benchmark = True
    cudnn.deterministic = True
    set_seed(args.seed)
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))
    if args.json is not None:
        logger.info('Overwriting config')
        overwrite_config(config, json.loads(args.json))

    device = torch.device(args.device)
    dataset_dict = util.get_all_datasets(config['datasets'])
    models_config = config['models']
    teacher_model_config = models_config.get('teacher_model', None)
    teacher_model = load_model(teacher_model_config, device) if teacher_model_config is not None else None
    student_model_config =\
        models_config['student_model'] if 'student_model' in models_config else models_config['model']
    ckpt_file_path = student_model_config.get('ckpt', None)
    student_model = load_model(student_model_config, device)
    total_adds, total_subs, total_mults, total_divs, total_exps, total_comps, total_ops, total_params = profile(student_model, (1,3,32,32))
    logger.info("================================================")
    logger.info("#Additions: %f Ops"%(total_adds))
    logger.info("#Subtractions: %f Ops"%(total_subs))
    logger.info("#Multiplications: %f Ops"%(total_mults))
    logger.info("#Divisions: %f Ops"%(total_divs))
    logger.info("#Exponentials: %f Ops"%(total_exps))
    logger.info("#Comparisons: %f Ops"%(total_comps))
    logger.info("#Ops: %f Ops"%(total_ops))
    logger.info("#Parameters: %f M"%(total_params))
    logger.info("================================================")
    # logger.info(student_model)
    if args.log_config:
        logger.info(config)

    if not args.test_only:
        train(teacher_model, student_model, dataset_dict, ckpt_file_path, device, device_ids, distributed, config, args)
        student_model_without_ddp =\
            student_model.module if module_util.check_if_wrapped(student_model) else student_model
        load_ckpt(student_model_config['ckpt'], model=student_model_without_ddp, strict=True)

    test_config = config['test']
    test_data_loader_config = test_config['test_data_loader']
    test_data_loader = util.build_data_loader(dataset_dict[test_data_loader_config['dataset_id']],
                                              test_data_loader_config, distributed)
    log_freq = test_config.get('log_freq', 1000)
    no_dp_eval = args.no_dp_eval
    iou_types = args.iou_types
    if not args.student_only and teacher_model is not None:
        evaluate(teacher_model, test_data_loader, iou_types, device, device_ids, distributed, no_dp_eval=no_dp_eval,
                 log_freq=log_freq, title='[Teacher: {}]'.format(teacher_model_config['name']))

    if check_if_updatable_detection_model(student_model):
        student_model.update()

    if check_if_analyzable(student_model):
        student_model.activate_analysis()
    evaluate(student_model, test_data_loader, iou_types, device, device_ids, distributed, no_dp_eval=no_dp_eval,
             log_freq=log_freq, title='[Student: {}]'.format(student_model_config['name']))


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())

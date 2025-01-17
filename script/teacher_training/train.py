from torchvision.models.mobilenetv3 import MobileNetV3, _mobilenet_v3_conf
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from cosine_scheduler import CosineWarmupLR
from torch.optim import lr_scheduler
from eval import accuracy
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import torchmetrics
# from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as trsf

import torch
import shutil
import os

def mixup_data(x, y, alpha):
    '''
    Returns mixed inputs, pairs of targets, and lambda
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint):
        print('**************')
        os.mkdir(checkpoint)

    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))



def get_transformer(mode: str, input_size = (32,32)):
    if mode == 'train':
        transform = trsf.Compose([
            trsf.ToTensor(),
            trsf.Normalize(mean= (0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
            trsf.RandomRotation(degrees=(-20,20)),
            trsf.RandomResizedCrop(input_size),
            trsf.RandomHorizontalFlip(),
        ])
    elif mode == 'test':
        transform = trsf.Compose([
            trsf.Resize((70, 70)),        
            trsf.CenterCrop((64, 64)),            
            trsf.ToTensor(),                
            trsf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    return transform

def adjust_learning_rate(optimizer, epoch):
    lr = optimizer.param_groups[0]['lr']

    if (epoch % 5) ==0:
        lr = lr**(epoch*0.4)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
def train(model, criterion, optimizer, dataloader:DataLoader, epoch:int, writer=None):
    model.train().to(device='cuda')

    prec1s=list()
    prec5s=list()
    losses = list()

    for i, (input, target) in enumerate(dataloader):

        # measure data loading time
        # data_time.update(time.time() - end)
        inputs = input.to('cuda')
        labels = target.to('cuda')

        # target = target.cuda(non_blocking=True)

        outputs = model(inputs)
        # input = input.to(device='cuda')
        inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, 0.2)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        acc1_a, acc5_a = accuracy(outputs, labels_a, topk=(1, 5))
        acc1_b, acc5_b = accuracy(outputs, labels_b, topk=(1, 5))
        # measure accuracy and record loss
        acc1 = lam * acc1_a + (1 - lam) * acc1_b
        acc5 = lam * acc5_a + (1 - lam) * acc5_b
        # compute output
        
        # print(f'output={output}')
        # loss = criterion(output, target)
        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output, target, topk=(1, 5))
        # losses.update(loss.item(), input.size(0))
        # top1.update(prec1.item(), input.size(0))
        # top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1s.append(acc1)
        prec5s.append(acc5)
        losses.append(loss)

    prec1_avg = sum(prec1s)/len(prec1s)
    prec5_avg = sum(prec5s)/len(prec5s)
    loss_avg = sum(losses)/len(losses)
    writer.add_scalar('train_loss', loss_avg, epoch+1)
    writer.add_scalar('train_prec1', prec1_avg, epoch+1)
    writer.add_scalar('train_prec5', prec5_avg, epoch+1)

    return loss_avg, prec1_avg, prec5_avg

def validate(model, val_loader, criterion):
def validate(model, val_loader, criterion):
    # switch to evaluate mode
    model.eval()#.to(device='cuda')
    prec1s=list()
    prec5s=list()
    losses=list()
    acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    for i, (input, target) in enumerate(val_loader):
        target = target.cpu()
        input = input.to(device='cpu')

        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)
            # output = torch.argmax(output, dim=1)
            
        print(acc(output, target))
        # prec1, prec5 = accuracy(output, target, topk=(1, 5))
        # print(output)
        # print(target)
    
        # prec1s.append(prec1)
        # prec5s.append(prec5)
        # losses.append(loss)

    prec1_avg = sum(prec1s)/len(prec1s)
    prec5_avg = sum(prec5s)/len(prec5s)
    loss_avg = sum(losses)/len(losses)

    # writer.add_scalar('val_loss', loss_avg, epoch+1)
    # writer.add_scalar('val_prec1', prec1_avg, epoch+1)
    # writer.add_scalar('val_prec5', prec5_avg, epoch+1)
    # writer.add_scalar('val_loss', loss_avg, epoch+1)
    # writer.add_scalar('val_prec1', prec1_avg, epoch+1)
    # writer.add_scalar('val_prec5', prec5_avg, epoch+1)

    return loss_avg, prec1_avg, prec5_avg

class Wrapper(torch.nn.Module):
    def __init__(self, original_model) -> None:
        super().__init__()
        self.model = deepcopy(original_model)
        self.unpack(original_model)

    def unpack(self, original_model):
        for idx, blocks in enumerate(original_model.features.children()):
            exec(f"self.model.features.layer{idx} = blocks")
    
    def forward(self, x):
        x = self.model(x)
        return x
        

def main():

    #grid search 
    PARAMS = {
    'start_lr': [0.12],
    'scheduler_step_size': [3],
    'gamma': [0.99],
    'weight_decay':  [6e-5]
    }

    my_configs = ParameterGrid(PARAMS)
    configurations = {"configurations": []}
    for config in my_configs:
        configurations["configurations"].append(config)

    print("Possible configurations found: {}".format(len(configurations["configurations"])))  

    for idx in tqdm(range(len(configurations["configurations"]))):

        config = configurations["configurations"][idx]
        formatted_config= f"step_{config['scheduler_step_size']}/gamma_{config['gamma']}/start_lr_{config['start_lr']}/weight_decay_{config['weight_decay']}/"
        print(formatted_config)
        other_formatting = f"step_{config['scheduler_step_size']}_gamma_{config['gamma']}_start_lr_{config['start_lr']}_weight_decay_{config['weight_decay']}"
        device = 'cpu'
        # model instantiation
        inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch='mobilenet_v3_small')
        model = MobileNetV3(inverted_residual_setting=inverted_residual_setting, last_channel=last_channel, num_classes=10)
        # dataloaders
        transformer = get_transformer('train')
        train_set = CIFAR10('~/dataset/cifar100', transform=transformer, download=True)
        train_loader = DataLoader(dataset=train_set, batch_size = 128, shuffle=True, pin_memory=True)

        transformer = get_transformer('test')
        val_set = CIFAR10('~/dataset/cifar100', transform=transformer, download=True, train=False)
        val_loader = DataLoader(dataset=val_set, batch_size = 128, shuffle=True, pin_memory=True)

        # loss setup
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.CrossEntropyLoss().cuda()

        # # optimizer 
        optimizer = torch.optim.SGD(model.parameters(), config['start_lr'],
                                    momentum=0.9,
                                    weight_decay=config['weight_decay'])
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=config['start_lr'], momentum=0.9,
        #                                 weight_decay=config['weight_decay'], eps=0.0316, alpha=0.9)
        
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step_size'], gamma=config['gamma'])
        scheduler = CosineWarmupLR(optimizer=optimizer, epochs=400, warmup_epochs=5, iter_in_one_epoch=len(train_loader))

    
        # train
        # writer = SummaryWriter("/home/g.esposito/AlternativeModels-SC2/script/teacher_training/400epochs/{}".format(formatted_config))
        best_prec1 = 0.0
        train_loss_avg=0.0
        val_loss_avg=0.0
        val_prec1_avg=0.0
        # writer.add_hparams(config,
        #                        {
        #                            'train_loss': train_loss_avg,
        #                            'val_loss': val_loss_avg,
        #                            'val_prec1_avg': val_prec1_avg
        #                        })

        # for epoch in tqdm(range(400)):
        #     print(f'epoch: {epoch}')
        #     train_loss_avg, train_prec1_avg, train_prec5_avg = train(model, criterion=criterion, optimizer = optimizer, dataloader=train_loader, epoch=epoch, writer=writer)
        # for epoch in tqdm(range(400)):
        #     print(f'epoch: {epoch}')
        #     train_loss_avg, train_prec1_avg, train_prec5_avg = train(model, criterion=criterion, optimizer = optimizer, dataloader=train_loader, epoch=epoch, writer=writer)

        #     val_loss_avg, val_prec1_avg, val_prec5_avg = validate(model, val_loader=val_loader, criterion=criterion, epoch=epoch, writer=writer)
        #     print(f'val_loss_avg: {val_loss_avg}, val_prec1_avg: {val_prec1_avg}, val_prec5_avg: {val_prec5_avg}')
        #     scheduler.step()
        #     val_loss_avg, val_prec1_avg, val_prec5_avg = validate(model, val_loader=val_loader, criterion=criterion, epoch=epoch, writer=writer)
        #     print(f'val_loss_avg: {val_loss_avg}, val_prec1_avg: {val_prec1_avg}, val_prec5_avg: {val_prec5_avg}')
        #     scheduler.step()

        #     is_best = val_prec1_avg > best_prec1
        #     best_prec1 = max(val_prec1_avg, best_prec1)
        #     is_best = val_prec1_avg > best_prec1
        #     best_prec1 = max(val_prec1_avg, best_prec1)

        #     save_checkpoint({
        #             'epoch': epoch + 1,
        #             'state_dict': model.state_dict(),
        #             'best_prec1': best_prec1,
        #             'optimizer': optimizer.state_dict(),
        #         }, is_best=is_best, checkpoint='/home/g.esposito/AlternativeModels-SC2/script/teacher_training/ckpt/mobilenet_cifar_SGD/{}'.format(other_formatting), filename='{:3f}_{}epoch.pth'.format(best_prec1, epoch))
        print(model)
        state_dict = torch.load('resource/ckpt/cifar10/teacher/cifar100-mobilenet_v3_small.ckpt')['state_dict']
        new_state_dict = dict()
        for key in state_dict.keys():
            fields = key.split('model.')
            new_k = fields[1]
            new_state_dict[new_k] = state_dict[key]
        # print(state_dict)
        model.load_state_dict(new_state_dict)
        val_loss_avg, val_prec1_avg, val_prec5_avg = validate(model, val_loader=val_loader, criterion=criterion)
        print(val_loss_avg)
        print(val_prec1_avg)
        print(val_prec5_avg)
        


if __name__ == '__main__':
    main()




# batch-size
# mode: using MobileNetV3-Small(if set to small) or MobileNetV3-Large(if set to large).
# dataset: which dataset to use(CIFAR10, CIFAR100, SVHN, TinyImageNet or ImageNet).
# ema-decay: decay of EMA, if set to 0, do not use EMA.
# label-smoothing: $epsilon$ using in label smoothing, if set to 0, do not use label smoothing.
# lr-decay: learning rate decay schedule, step or cos.
# lr-min: min lr in cos lr decay.
# warmup-epochs: warmup epochs using in cos lr deacy.
# num-epochs: total training epochs.
# nbd: no bias decay.
# zero-gamma: zero $gamma$ of last BN in each block.
# mixup: using Mixup.



    # parser.add_argument('--num-class', type=int, default=1000)
    # parser.add_argument('--label-smoothing', type=float, default=0.1, help='label smoothing') # 0.1
    # parser.add_argument('--decay-rate', type=float, default=1, help='decay rate in CosineAnnealingWarmRestarts')
    # parser.add_argument('--bn-momentum', type=float, default=0.1, help='momentum in BatchNorm2d')
    # parser.add_argument('-nbd', default=False, action='store_true', help='no bias decay') # True
    # parser.add_argument('-zero-gamma', default=False, action='store_true', help='zero gamma in BatchNorm2d when init') # True
    # parser.add_argument('-mixup', default=False, action='store_true', help='mixup or not') # True
    # parser.add_argument('--mixup-alpha', type=float, default=0.2, help='alpha used in mixup')


# CUDA_VISIBLE_DEVICES=1 python train.py --batch-size=128 --mode=small --print-freq=100 --dataset=CIFAR100\
#   --ema-decay=0 --label-smoothing=0 --lr=0.35 --save-epoch-freq=1000 --lr-decay=cos --lr-min=0\
#   --warmup-epochs=5 --weight-decay=6e-5 --num-epochs=400 --num-workers=2 --width-multiplier=1

# CUDA_VISIBLE_DEVICES=1 python train.py --batch-size=128 --mode=small --print-freq=100 --dataset=CIFAR100\
#   --ema-decay=0.999 --label-smoothing=0.1 --lr=0.35 --save-epoch-freq=1000 --lr-decay=cos --lr-min=0\
#   --warmup-epochs=5 --weight-decay=6e-5 --num-epochs=400 --num-workers=2 --width-multiplier=1\
#   -zero-gamma -nbd -mixup
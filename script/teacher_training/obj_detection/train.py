from torchvision.models.detection.ssd import SSD, vgg16, DefaultBoxGenerator, VGG16_Weights, _vgg_extractor
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from eval import compute_mAP
from tqdm import tqdm
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as trsf

import torch
import shutil
import os

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
            # you can add other transformations in this list
            # trsf.Resize(input_size),
            trsf.ToTensor(),
            trsf.Normalize(mean= (0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
        ])
    return transform

    
def train(model, optimizer, dataloader:DataLoader, epoch:int, writer=None):
    model.train().to(device='cuda')

    losses = list()

    for i, (input, target) in enumerate(dataloader):

        # measure data loading time
        # data_time.update(time.time() - end)

        # target = target.cuda(non_blocking=True)
        print(target)
        input = input.to(device='cuda')
        # compute output
        loss = model(input, target)
        # print(f'output={output}')
        # measure accuracy and record loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss)

    loss_avg = sum(losses)/len(losses)
    writer.add_scalar('train_loss', loss_avg, epoch+1)

    return loss_avg

def validate(model, val_loader, criterion, epoch, writer=None):
    # switch to evaluate mode
    model.eval().to(device='cuda')
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.to(device='cuda')

        with torch.no_grad():
            # compute output
            output = model(input)

            print(output)

    # writer.add_scalar('val_loss', loss_avg, epoch+1)

    # return loss_avg, prec1_avg, prec5_avg

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
    'start_lr': [0.1],
    'scheduler_step_size': [20, 30],
    'gamma': [0.1, 0.05],
    'weight_decay':  [1e-4]
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

        # model instantiation
        size=(300,300)
        
        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES, progress=True)
        backbone=_vgg_extractor(backbone, False, 4)
        anchor_generator = DefaultBoxGenerator(
            [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
            steps=[8, 16, 32, 64, 100, 300],
        )
        image_mean=[0.48235, 0.45882, 0.40784]
        image_std=[1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]

        model = SSD(backbone=backbone, anchor_generator=anchor_generator, size=size, image_mean=image_mean, image_std=image_std, num_classes=91)
        
        # dataloaders
        transformer = get_transformer('train')
        train_set = CocoDetection('~/dataset/coco2017/train2017',annFile='/home/g.esposito/dataset/coco2017/annotations/person_keypoints_train2017.json', transform=transformer)
        train_loader = DataLoader(dataset=train_set, batch_size = 1, shuffle=True, pin_memory=True)

        transformer = get_transformer('test')
        val_set = CocoDetection('~/dataset/coco2017/val2017',annFile='/home/g.esposito/dataset/coco2017/annotations/person_keypoints_val2017.json', transform=transformer)
        val_loader = DataLoader(dataset=val_set, batch_size = 1, shuffle=True, pin_memory=True)


        # optimizer 
        optimizer = torch.optim.SGD(model.parameters(), config['start_lr'],
                                    momentum=0.9,
                                    weight_decay=config['weight_decay'])
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=config['start_lr'], momentum=0.9,
        #                                 weight_decay=config['weight_decay'], eps=0.0316, alpha=0.9)
        
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step_size'], gamma=config['gamma'])

    
        # train
        writer = SummaryWriter("/home/g.esposito/AlternativeModels-SC2/script/teacher_training/obj_detection/100epochs/{}".format(formatted_config))
        best_prec1 = 0.0


        for epoch in tqdm(range(150)):
            print(f'epoch: {epoch}')
            train_loss_avg, train_prec1_avg, train_prec5_avg = train(model, optimizer = optimizer, dataloader=train_loader, epoch=epoch, writer=writer)

            val_loss_avg, val_prec1_avg, val_prec5_avg = validate(model, val_loader=val_loader, epoch=epoch, writer=writer)
            print(f'val_loss_avg: {val_loss_avg}, val_prec1_avg: {val_prec1_avg}, val_prec5_avg: {val_prec5_avg}')
            scheduler.step()

            is_best = val_prec1_avg > best_prec1
            best_prec1 = max(val_prec1_avg, best_prec1)

            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }, is_best=is_best, checkpoint='/home/g.esposito/AlternativeModels-SC2/script/teacher_training/obj_detection/ckpt/mobilenet_cifar_SGD/{}'.format(other_formatting), filename='{}.pth'.format(best_prec1))


if __name__ == '__main__':
    main()
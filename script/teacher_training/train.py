from torchvision.models.mobilenetv3 import MobileNetV3, _mobilenet_v3_conf
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from torch.optim import lr_scheduler
from eval import accuracy
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

        target = target.cuda(non_blocking=True)
        input = input.to(device='cuda')
        # compute output
        output = model(input)
        # print(f'output={output}')
        loss = criterion(output, target)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        # losses.update(loss.item(), input.size(0))
        # top1.update(prec1.item(), input.size(0))
        # top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1s.append(prec1)
        prec5s.append(prec5)
        losses.append(loss)

    prec1_avg = sum(prec1s)/len(prec1s)
    prec5_avg = sum(prec5s)/len(prec5s)
    loss_avg = sum(losses)/len(losses)
    writer.add_scalar('train_loss', loss_avg, epoch+1)
    writer.add_scalar('train_prec1', prec1_avg, epoch+1)
    writer.add_scalar('train_prec5', prec5_avg, epoch+1)

    return loss_avg, prec1_avg, prec5_avg

def validate(model, val_loader, criterion, epoch, writer=None):
    # switch to evaluate mode
    model.eval().to(device='cuda')
    prec1s=list()
    prec5s=list()
    losses=list()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.to(device='cuda')

        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
    
        prec1s.append(prec1)
        prec5s.append(prec5)
        losses.append(loss)

    prec1_avg = sum(prec1s)/len(prec1s)
    prec5_avg = sum(prec5s)/len(prec5s)
    loss_avg = sum(losses)/len(losses)

    writer.add_scalar('val_loss', loss_avg, epoch+1)
    writer.add_scalar('val_prec1', prec1_avg, epoch+1)
    writer.add_scalar('val_prec5', prec5_avg, epoch+1)

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
        inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch='mobilenet_v3_small')
        model = MobileNetV3(inverted_residual_setting=inverted_residual_setting, last_channel=last_channel)
        # dataloaders
        transformer = get_transformer('train')
        train_set = CIFAR100('~/dataset/cifar100', train=True, transform=transformer, download=True)
        train_loader = DataLoader(dataset=train_set, batch_size = 2048, shuffle=True, pin_memory=True)

        transformer = get_transformer('test')
        val_set = CIFAR100('~/dataset/cifar100', transform=transformer, download=True)
        val_loader = DataLoader(dataset=val_set, batch_size = 2048, shuffle=True, pin_memory=True)

        # loss setup
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.CrossEntropyLoss().cuda()

        # optimizer 
        optimizer = torch.optim.SGD(model.parameters(), config['start_lr'],
                                    momentum=0.9,
                                    weight_decay=config['weight_decay'])
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=config['start_lr'], momentum=0.9,
        #                                 weight_decay=config['weight_decay'], eps=0.0316, alpha=0.9)
        
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step_size'], gamma=config['gamma'])

    
        # train
        writer = SummaryWriter("/home/g.esposito/AlternativeModels-SC2/script/teacher_training/100epochs/{}".format(formatted_config))
        best_prec1 = 0.0
        # train_loss_avg=0.0
        # val_loss_avg=0.0
        # val_prec1_avg=0.0
        # writer.add_hparams(config,
        #                        {
        #                            'train_loss': train_loss_avg,
        #                            'val_loss': val_loss_avg,
        #                            'val_prec1_avg': val_prec1_avg
        #                        })

        for epoch in tqdm(range(150)):
            print(f'epoch: {epoch}')
            train_loss_avg, train_prec1_avg, train_prec5_avg = train(model, criterion=criterion, optimizer = optimizer, dataloader=train_loader, epoch=epoch, writer=writer)

            val_loss_avg, val_prec1_avg, val_prec5_avg = validate(model, val_loader=val_loader, criterion=criterion, epoch=epoch, writer=writer)
            print(f'val_loss_avg: {val_loss_avg}, val_prec1_avg: {val_prec1_avg}, val_prec5_avg: {val_prec5_avg}')
            scheduler.step()

            is_best = val_prec1_avg > best_prec1
            best_prec1 = max(val_prec1_avg, best_prec1)

            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }, is_best=is_best, checkpoint='/home/g.esposito/AlternativeModels-SC2/script/teacher_training/ckpt/mobilenet_cifar_SGD/{}'.format(other_formatting), filename='{}.pth'.format(best_prec1))


if __name__ == '__main__':
    main()
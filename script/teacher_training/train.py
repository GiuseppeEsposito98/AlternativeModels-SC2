from torchvision.models.mobilenetv3 import MobileNetV3, _mobilenet_v3_conf
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from eval import accuracy

import torchvision.transforms as trsf

import torch



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

    if epoch == 5 or epoch == 10 or epoch == 15 or epoch == 20:
        lr = lr^(epoch*0.4)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
def train(model, criterion, optimizer, dataloader:DataLoader, epoch:int):
    model.train()

    prec1s=list()
    prec5s=list()
    losses = list()

    for i, (input, target) in enumerate(dataloader):
        adjust_learning_rate(optimizer, epoch)

        # measure data loading time
        # data_time.update(time.time() - end)

        # target = target.cuda(non_blocking=True)
        # compute output
        output = model(input)
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

    return loss_avg, prec1_avg, prec5_avg

def validate(model, val_loader, criterion):
    # switch to evaluate mode
    model.eval()
    prec1s=list()
    prec5s=list()
    losses=list()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)

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

    return loss_avg, prec1_avg, prec5_avg

def main():
    # model instantiation
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch='mobilenet_v3_small')
    model = MobileNetV3(inverted_residual_setting=inverted_residual_setting, last_channel=last_channel)

    # dataloaders
    transformer = get_transformer('train')
    train_set = CIFAR100('~/dataset/cifar100', train=True, transform=transformer)
    train_loader = DataLoader(dataset=train_set, batch_size = 16, shuffle=True, pin_memory=True)

    transformer = get_transformer('test')
    val_set = CIFAR100('~/dataset/cifar100', transform=transformer)
    val_loader = DataLoader(dataset=val_set, batch_size = 16, shuffle=True, pin_memory=True)

    # loss setup
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.CrossEntropyLoss().cuda()

    # optimizer 
    optimizer = torch.optim.SGD(model.parameters(), 0.001,
                                momentum=0.9,
                                weight_decay=1e-4)

    # train
    for epoch in range(20):
        train_loss_avg, train_prec1_avg, train_prec5_avg = train(model, criterion=criterion, optimizer = optimizer, dataloader=train_loader, epoch=epoch)

        val_loss_avg, val_prec1_avg, val_prec5_avg = validate(model, val_loader=val_loader, criterion=criterion)
        print(f'val_loss_avg: {val_loss_avg}, val_prec1_avg: {val_prec1_avg}, val_prec5_avg: {val_prec5_avg}')

    # eval

if __name__ == '__main__':
    main()
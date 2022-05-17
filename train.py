import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from logger import Logger
from networks.wideresnet import *

def func(epoch):
    if epoch < 79:
        return 1.0
    elif epoch < 139:
        return 0.1
    elif epoch < 179:
        return 0.1**2
    else:
        return 0.1**3

######################################
# Custom Dataloader
######################################
class Cat_dataloader(Dataset):
    def __init__(self, data, is_train=True, transform=None):
        self.data = datasets.__dict__[data.upper()]('./data', train=is_train, download=True, transform=transform)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        image = self.data[i][0]
        label = self.data[i][1]
        index = i
        return image, label, index

#########################################
# Label Smoothing using Dirichlet dist.
#########################################
def label_smoothing(targets, epsilon, c, num_classes=10):
    onehot = torch.eye(num_classes)[targets].cuda()
    dirich = torch.from_numpy(np.random.dirichlet(np.ones(num_classes), targets.size(0))).cuda()
    sr = (torch.ones(targets.size(0)).cuda() * (c*epsilon)).unsqueeze(1).repeat(1, num_classes)
    ones = torch.ones_like(sr)
    y_tilde = (ones - sr) * onehot + sr * dirich
    return y_tilde

#########################################
# Loss functions
#########################################
class Loss_func():
    def __init__(self, num_classes=10, loss_type='xent'):
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.loss_type = loss_type
        
    def xent(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        loss = -torch.sum(targets * torch.log(probs))/probs.size(0)
        return loss
    
    def mix(self, logits, targets, org_targets):
        batch = probs.size(0)
        class_index = torch.arange(num_classes)[None,:].repeat(batch,1).cuda()
        false_probs = torch.topk(probs[class_index!=targets[:,None]].view(batch, num_classes-1), k=1).values
        gt_probs = probs[class_index==targets[:,None]].unsqueeze(1)
        cw_loss = false_probs - gt_probs
        loss = torch.sum(torch.sum(-targets * torch.log(probs), dim=1) + cw_loss)/probs.size(0)
        return loss
    
    def __call__(self, logits, targets, org_targets):
        if self.loss_type == 'xent':
            return self.xent(logits, targets)
        elif self.loss_type == 'mix':
            return self.mix(logits, targets, org_targets)

#########################################
# Inner Maximization
#########################################
def inner_maximization(model, loss_func, inputs, targets, org_targets, epsilons, alpha, num_steps):
    epsilons = epsilons[:,None,None,None].repeat(1, inputs.size(1), inputs.size(2), inputs.size(3))
    for _ in range(num_steps):
        x = inputs.requires_grad_()
        logits = model(x)
        loss = loss_func(logits, targets, org_targets)
        loss.backward()
        grads = x.grad.data
        x = x.data.detach() + alpha * torch.sign(grads).detach()
        x = torch.min(torch.max(x, inputs-epsilons), inputs+epsilons)
        x = torch.clamp(x, min=0, max=1)
    return x

def train(epoch, model, dataloader, optimizer, num_classes, loss_func, epsilons, alpha, num_steps, c, eta, epsilon_max):
    model.train()
    
    for idx, samples in enumerate(dataloader):
        start_time = time.time()
        inputs, targets, indices = samples
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        smoothed_targets = label_smoothing(targets, epsilons[indices], c, num_classes)
        epsilons[indices] += eta
        x = inner_maximization(model, loss_func, inputs, smoothed_targets, targets, 
                               epsilons[indices], alpha, num_steps)
        
        logits = model(x)
        t_or_f = torch.argmax(torch.softmax(logits, dim=1), dim=1).eq(targets)
        false_indices = indices[torch.where(t_or_f==False)[0]]
        epsilons[false_indices] -= eta
        epsilons[indices] = torch.min(epsilons[indices], (torch.ones(inputs.size(0)) * epsilon_max).cuda())
        
        smoothed_targets = label_smoothing(targets, epsilons[indices], c, num_classes)
        loss = loss_func(logits, smoothed_targets, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        num_correct = torch.argmax(torch.softmax(logits, dim=1), dim=1).eq(targets).sum().item()
        train_rob_acc = 100 * (num_correct/inputs.size(0))
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        if idx % 100 == 0:
            print('Epoch %d [%d/%d] | loss: %.4f | rob acc: %.4f | elapsed time: %.4f' % (epoch, idx+1, 
                                                                                          len(dataloader), 
                                                                                          loss.item(), 
                                                                                          train_rob_acc,
                                                                                          elapsed_time))
        
def evaluation(epoch, model, dataloader, loss_func, epsilon, alpha, num_steps):
    model.eval()
    counter = 0 
    total_corr_nat = 0
    total_corr_rob = 0
    xent = nn.CrossEntropyLoss()
    
    for idx, samples in enumerate(dataloader):
        inputs, targets, _ = samples
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        total_corr_nat += torch.argmax(torch.softmax(model(inputs), dim=1), dim=1).eq(targets).sum().item()
        
        noise = torch.FloatTensor(inputs.size()).uniform_(-epsilon, epsilon).cuda()
        x = torch.clamp(inputs + noise, min=0, max=1)
        for _ in range(num_steps):
            x.requires_grad_()
            logits = model(x)
            loss = xent(logits, targets)
            loss.backward()
            grads = x.grad.data
            x = x.data.detach() + alpha * torch.sign(grads).detach()
            x = torch.min(torch.max(x, inputs-epsilon), inputs+epsilon)
            x = torch.clamp(x, min=0, max=1)
            
        total_corr_rob += torch.argmax(torch.softmax(model(x), dim=1), dim=1).eq(targets).sum().item()
        counter += inputs.size(0)
        
        sys.stdout.write('\r [Eval] [{}/{} ({:.0f}%)]\tacc nat: {:.5f}[%]\tacc rob: {:.5f}[%]'.format(
            idx*len(inputs), len(dataloader.dataset), 100.*idx/len(dataloader),
            100*(total_corr_nat/counter), 100*(total_corr_rob/counter)))
        
    avg_nat = 100 * (total_corr_nat/len(dataloader.dataset))
    avg_rob = 100 * (total_corr_rob/len(dataloader.dataset))
    print()
    return avg_nat, avg_rob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_torch', type=int, default=np.random.randint(4294967295))
    parser.add_argument('--seed_numpy', type=int, default=np.random.randint(4294967295))
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default='./checkpoint')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'])
    parser.add_argument('--loss_type', type=str, choices=['xent', 'mix'])
    ## Inner maximization
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=2)
    parser.add_argument('--epsilon', type=float, default=8)
    parser.add_argument('--eta', type=float, default=5e-3)
    parser.add_argument('--c', type=float, default=10)
    parser.add_argument('--kappa', type=float, default=10)
    parser.add_argument('--epsilon_max', type=float, default=8)
    ## Optimizer
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    
    np.random.seed(args.seed_numpy)
    torch.manual_seed(args.seed_torch)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    out_dir = os.path.join(args.checkpoint, args.dataset, args.loss_type)
    os.makedirs(out_dir, exist_ok=True)
    
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    train_dataset = Cat_dataloader(args.dataset, is_train=True, transform=train_transforms)
    test_dataset = Cat_dataloader(args.dataset, is_train=False, transform=transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  shuffle=True, drop_last=False, num_workers=os.cpu_count())
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                 shuffle=False, drop_last=False, num_workers=os.cpu_count())
    
    model = nn.DataParallel(WideResNet(depth=34, num_classes=args.num_classes, widen_factor=10, dropRate=0.0).cuda())
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    adjust_lr = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
    
    loss_func = Loss_func(num_classes=args.num_classes, loss_type=args.loss_type)
    logger_test = Logger(os.path.join(out_dir, 'log_results.txt'))
    logger_test.set_names(['Epoch', 'Natural Test Acc', 'PGD20 Acc'])
    
    best_acc = 0
    epsilons = torch.zeros(len(train_dataloader.dataset)).cuda()
    for epoch in range(args.epochs):
        train(epoch, model, train_dataloader, optimizer, args.num_classes, loss_func,
              epsilons, args.alpha/255, args.num_steps, args.c, args.eta, args.epsilon_max/255)
        avg_nat, avg_rob = evaluation(epoch, model, test_dataloader, loss_func, 
                                      args.epsilon/255, args.alpha/255, 20)
        logger_test.append([epoch+1, avg_nat, avg_rob])
        
        if avg_rob > best_acc:
            best_acc = avg_rob
            best_checkpoint = {'epoch': epoch+1,
                               'state_dict': model.state_dict(),
                               'test_nat_acc': avg_nat,
                               'test_rob_acc': avg_rob,
                               'optimizer': optimizer.state_dict()}
            torch.save(best_checkpoint, os.path.join(out_dir, 'bestpoint.pth.tar'))
            
        std_checkpoint = {'epoch': epoch+1,
                          'state_dict': model.state_dict(),
                          'test_nat_acc': avg_nat,
                          'test_rob_acc': avg_rob,
                          'optimizer': optimizer.state_dict()}
        torch.save(std_checkpoint, os.path.join(out_dir, 'std_checkpoint.pth.tar'))
        adjust_lr.step()
    
    
    
    
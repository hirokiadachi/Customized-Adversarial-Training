import os
import numpy as np
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from advertorch.utils import CarliniWagnerLoss
from advertorch.attacks import CarliniWagnerL2Attack
from advertorch.attacks import FGSM, LinfPGDAttack, L2PGDAttack
from autoattack import AutoAttack
from cw_attack import CarliniWagnerLIAttack

from networks.wideresnet import *

def natural_acc(model, dataloader):
    model.eval()
    total_correct = 0
    
    for (inputs, targets) in tqdm(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        
        with torch.no_grad():
            logits = model(inputs)
        total_correct += torch.softmax(logits, dim=1).argmax(dim=1).eq(targets).sum().item()
    
    avg_acc = total_correct*100/len(dataloader.dataset)
    print('Natural acc: %.4f' % avg_acc)
    
class EvalRobustness():
    def __init__(self, model, dataloader, epsilon, alpha, lower, upper):
        self.model = model
        self.dataloader = dataloader
        self.eps = epsilon
        self.alpha = alpha
        self.lower = lower
        self.upper = upper
        self.xent = nn.CrossEntropyLoss()
        
    def fgsm(self, ):
        total_correct = 0
        fgsm = FGSM(predict=self.model, 
                    loss_fn=self.xent, 
                    eps=self.eps, 
                    clip_min=self.lower, 
                    clip_max=self.upper, 
                    targeted=False)
        
        for inputs, targets in tqdm(self.dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            x = fgsm(inputs, targets)
            with torch.no_grad():
                self.model.eval()
                logits = self.model(x)
                
            total_correct += torch.softmax(logits, dim=1).argmax(dim=1).eq(targets).sum().item()
        
        avg_acc = total_correct*100/len(self.dataloader.dataset)
        print('Robust acc (FGSM): %.4f' % avg_acc)
        
        
    
    def pgd(self, num_steps, norm='linf'):
        total_correct = 0
        if norm == 'linf':
            pgd = LinfPGDAttack(predict=self.model,
                                loss_fn=self.xent,
                                eps=self.eps,
                                nb_iter=num_steps,
                                eps_iter=self.alpha,
                                rand_init=True,
                                clip_min=self.lower,
                                clip_max=self.upper,
                                targeted=False)
        elif norm == 'l2':
            pgd = L2PGDAttack(predict=self.model,
                              loss_fn=self.xent,
                              eps=self.eps,
                              nb_iter=num_steps,
                              eps_iter=self.alpha,
                              rand_init=True,
                              clip_min=self.lower,
                              clip_max=self.upper,
                              targeted=False)
        else:
            assert 0, 'Error: %s norm is not supported.' % norm
        
        for inputs, targets in tqdm(self.dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            x = pgd(inputs, targets)
        
            with torch.no_grad():
                self.model.eval()
                logits = self.model(x)
        
            total_correct += torch.softmax(logits, dim=1).argmax(dim=1).eq(targets).sum().item()
    
        avg_acc = total_correct*100/len(self.dataloader.dataset)
        print('Robust acc (PGD-%d): %.4f' % (num_steps, avg_acc))
    
    def pgd_with_cw_loss(self, num_steps, norm='linf'):
        total_correct = 0
        if norm == 'linf':
            pgd = LinfPGDAttack(predict=self.model,
                                loss_fn=CarliniWagnerLoss(),
                                eps=self.eps,
                                nb_iter=num_steps,
                                eps_iter=self.alpha,
                                rand_init=True,
                                clip_min=self.lower,
                                clip_max=self.upper,
                                targeted=False)
        elif norm == 'l2':
            pgd = L2PGDAttack(predict=self.model,
                              loss_fn=CarliniWagnerLoss(),
                              eps=self.eps,
                              nb_iter=num_steps,
                              eps_iter=self.alpha,
                              rand_init=True,
                              clip_min=self.lower,
                              clip_max=self.upper,
                              targeted=False)
        else:
            assert 0, "Error: %s norm is not supported." % norm
            
        for inputs, targets in tqdm(self.dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            x = pgd(inputs, targets)
            
            with torch.no_grad():
                self.model.eval()
                logits = self.model(x)
                
            total_correct += logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        
        avg_acc = total_correct/len(self.dataloader.dataset)
        print('Avg acc (PGD-%d w/ cw loss): %.4f' % (num_steps, avg_acc))
    
    def cw_l2(self, num_classes):
        total_correct = 0
        cw = CarliniWagnerL2Attack(predict=self.model, 
                                   num_classes=num_classes,
                                   confidence=0, 
                                   targeted=False, 
                                   learning_rate=5e-3,
                                   binary_search_steps=5, 
                                   max_iterations=20, 
                                   abort_early=True, 
                                   initial_const=1e-2, 
                                   clip_min=self.lower, 
                                   clip_max=self.upper, 
                                   loss_fn=None)
        
        for inputs, targets in tqdm(self.dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            x = cw.perturb(inputs, targets)
            delta = x.data - inputs.data
            x = torch.clamp(inputs.data+delta, min=self.lower, max=self.upper)
            
            with torch.no_grad():
                self.model.eval()
                logits = self.model(x)
                
            total_correct += torch.softmax(logits, dim=1).argmax(dim=1).eq(targets).sum().item()
        
        avg_acc = total_correct*100/len(self.dataloader.dataset)
        print('Robust acc (CW-l2): %.4f' % (avg_acc))
    
    def apgd_ce(self):
        total_correct = 0
        AA = AutoAttack(self.model, norm='Linf', eps=self.eps, version='standard')
        
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            AA.attacks_to_run = ['apgd-ce']
            x = AA.run_standard_evaluation_individual(inputs, targets, bs=inputs.size(0))['apgd-ce']
            with torch.no_grad():
                self.model.eval()
                out = self.model(x)
            
            total_correct += torch.argmax(torch.softmax(out, dim=1), dim=1).eq(targets).sum().item()
            
        avg_acc = total_correct/len(self.dataloader.dataset)
        print('Avg acc (APGD-CE): %.4f' % avg_acc) 

    def apgd_dlr(self):
        total_correct = 0
        AA = AutoAttack(self.model, norm='Linf', eps=self.eps, version='standard')
        
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            AA.attacks_to_run = ['apgd-t']
            x = AA.run_standard_evaluation_individual(inputs, targets, bs=inputs.size(0))['apgd-t']
            with torch.no_grad():
                self.model.eval()
                out = self.model(x)
            
            total_correct += torch.argmax(torch.softmax(out, dim=1), dim=1).eq(targets).sum().item()
            
        avg_acc = total_correct/len(self.dataloader.dataset)
        print('Avg acc (APGD-DLR): %.4f' % avg_acc)
    
    def aa(self):
        total_correct = 0
        AA = AutoAttack(self.model, norm='Linf', eps=self.eps, version='standard')
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            x = AA.run_standard_evaluation(inputs, targets, bs=inputs.size(0))
            
            with torch.no_grad():
                self.model.eval()
                logits = self.model(x)
                
            total_correct += logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
            
        avg_acc = total_correct/len(self.dataloader.dataset)
        print('Avg acc (AA): %.4f' % avg_acc)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint')
    parser.add_argument('--norm', type=str, default='linf')
    parser.add_argument('--eps', type=int, default=8)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--nat', action='store_true')
    parser.add_argument('--fgsm', action='store_true')
    parser.add_argument('--pgd', action='store_true')
    parser.add_argument('--cw', action='store_true')
    parser.add_argument('--cw_pgd', action='store_true')
    parser.add_argument('--aa', action='store_true')
    parser.add_argument('--apgd_ce', action='store_true')
    parser.add_argument('--apgd_dlr', action='store_true')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    checkpoint = torch.load(args.checkpoint)
    state_dict = checkpoint['state_dict']
    num_classes = args.num_classes
    
    model = nn.DataParallel(WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0).cuda())
    #model = nn.DataParallel(resnet.ResNet18(num_classes=num_classes).cuda())
    model.load_state_dict(state_dict)
    model.eval()
    
    try:
        dataset = datasets.__dict__[args.dataset.upper()]('./data', download=True, train=False, transform=transforms.ToTensor())
    except:
        dataset = datasets.__dict__[args.dataset.upper()]('./data', download=True, split='test', transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=100, drop_last=False, shuffle=False)
    
    lower, upper = 0, 1
    epsilon = args.eps/255
    alpha = epsilon/5
    num_steps_list = [10, 20, 100]
    
    evalator = EvalRobustness(model, dataloader, epsilon, alpha, lower, upper)
    if args.nat:
        natural_acc(model, dataloader)
    
    if args.fgsm:
        evalator.fgsm()
    
    if args.pgd:
        for n in num_steps_list:
            print('num PGD steps: %d' % n)
            evalator.pgd(n)
    
    if args.cw_pgd:
        evalator.pgd_with_cw_loss(num_steps=100)    
    
    if args.cw:
        evalator.cw_l2(num_classes)
    
    if args.apgd_ce:
        evalator.apgd_ce()
    
    if args.apgd_dlr:
        evalator.apgd_dlr()
    
    if args.aa:
        evalator.aa()
        
    
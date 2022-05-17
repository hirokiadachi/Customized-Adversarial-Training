# Customized-Adversarial-Training
Pytorch implementation of "CAT: Customized Adversarial Training for Improved Robustness".<br>
Authors: Minhao Cheng, Qi Lei, Pin-Yu Chen, Inderjit Dhillon, and Cho-Jui Hsieh<br>
Paper: https://arxiv.org/abs/2002.06789

## Training details
Model: WideResNet34-10<br>
Epochs: 200<br>
Batch size: 128<br>
Optimizer: SGD
 - momentum: 0.9
 - initial learning rate: 0.1
   - 80th: 0.01
   - 140th: 0.001
   - 180th: 0.0001

* CIFAR10, CrossEntropyLoss
```
python train.py --dataset cifar10 --num_classes 10 --loss_type xent --gpus 0
```

|     |Natural|FGSM|PGD-10|PGD-20|PGD-100|PGD-100(CW loss)|CW-20 (l2)|APGD-CE|APGD-DLR|
|:---:|:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|CIFAR-10, Xent|93.73|78.86|71.53|68.12|64.36|64.83|68.71|56.12||

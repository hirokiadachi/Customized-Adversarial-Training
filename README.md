# Customized-Adversarial-Training
Pytorch implementation of "CAT: Customized Adversarial Training for Improved Robustness".<br>
Authors: Minhao Cheng, Qi Lei, Pin-Yu Chen, Inderjit Dhillon, and Cho-Jui Hsieh<br>
Paper: https://arxiv.org/abs/2002.06789

## Training details
Model: WideResNet34-10<br>
Epochs: 200<br>
Batch size: 128<br>
Optimizer: SGD<br>
(momentum: 0.9)<br>
(initial learning rate: 0.1)<br>
(80th: 0.01)<br>
(140th: 0.001)<br>
(180th: 0.0001)<br>

* CIFAR10
```
python train.py --dataset cifar10 --num_classes 10 --loss_type xent --gpus 0
python train.py --dataset cifar10 --num_classes 10 --loss_type mix --gpus 0
```
* CIFAR100
```
python train.py --dataset cifar100 --num_classes 100 --loss_type xent --gpus 0
python train.py --dataset cifar100 --num_classes 100 --loss_type mix --gpus 0
```

|     |Natural|FGSM|PGD-10|PGD-20|PGD-100|PGD-100(CW loss)|CW-20 (l2)|APGD-CE|APGD-DLR|
|:---:|:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|CIFAR-10, Xent|93.73|78.86|71.53|68.12|64.36|64.83|68.71|56.12|23.91|
|CIFAR-10, Mix |93.66|80.57|73.80|68.61|62.38|62.50|73.30|53.85|26.16| 
|CIFAR-100, Xent|72.14|46.64|40.79|39.42|37.80|20.17|26.04|34.80|13.17|

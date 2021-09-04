# Record
----------
## Optimizer

可参考知乎网址：[优化器](https://zhuanlan.zhihu.com/p/78622301)

一般使用以下两个优化器
1. SGD
2. Adam
3. RMSProp

### SGD
     optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
        
使用SGD的优化器，第一个参数是要用这个优化器优化的权重参数的key，后面是学习率和momentum动量。
Momentum(动量、冲量)：结合当前梯度与上一次更新信息，用于当前更新,表示上一更新信息留到这次更新的权重比例问题。
### Adam
     torch.optim.Adam(params,lr=0.001,betas=(0.9, 0.999),eps=1e-08,weight_decay=0,amsgrad=False)

在RMSProp的基础上，做两个改进：梯度滑动平均和偏差纠正。

### RMSProp

     torch.optim.RMSprop(params,lr=0.01,alpha=0.99,eps=1e-08,weight_decay=0,momentum=0,centered=False)
RMSProp算法通过累计各个变量的梯度的平方r，然后用每个变量的梯度除以r，即可有效缓解变量间的梯度差异。

## lr_scheduler

可参考CSDN的博客[学习率](https://blog.csdn.net/shanglianlm/article/details/85143614)
1. StepLR
2. MultiStepLR
3. ExponentialLR
4. CosineAnnealingLR
5. ReduceLROnPlateau
6. LambdaLR
### StepLR
等间隔调整学习率，调整倍数为 gamma 倍，调整间隔为 step_size。间隔单位是step。需要注意的是， step 通常是指 epoch，不要弄成 iteration 了。

    torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
    
1. step_size(int)- 学习率下降间隔数，若为 30，则会在 30、 60、 90…个 step 时，将学习率调整为 lr*gamma。
2. gamma(float)- 学习率调整倍数，默认为 0.1 倍，即下降 10 倍。
3. last_epoch(int) 上一个 epoch 数，这个变量用来指示学习率是否需要调整。当last_epoch 符合设定的间隔时，就会对学习率进行调整。当为-1 时，学习率设置为初始值


### MultiStepLR

按设定的间隔调整学习率。这个方法适合后期调试使用，观察 loss 曲线，为每个实验定制学习率调整时机。

    torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
1. milestones(list)- 一个 list，每一个元素代表何时调整学习率， list 元素必须是递增的。如 milestones=[30,80,120]
2. gamma(float)- 学习率调整倍数，默认为 0.1 倍，即下降 10 倍。

### ExponentialLR

按指数衰减调整学习率：

    torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)

1. gamma 学习率调整倍数的底，指数为 epoch，即 gamma**epoch

### CosineAnnealingLR
让学习率随epoch的变化图类似于cos：

    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)

1. 余弦退火学习率中LR的变化是周期性的，T_max是周期的1/2
2. eta_min 是指周期中最小的学习率。

### ReduceLROnPlateau
当某指标不再变化（下降或升高），调整学习率，这是非常实用的学习率调整策略。
例如，当验证集的 loss 不再下降时，进行学习率调整；或者监测验证集的 accuracy，当accuracy 不再上升时，则调整学习率。

    torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08

1. mode(str)- 模式选择，有 min 和 max 两种模式， min 表示当指标不再降低(如监测loss)， max 表示当指标不再升高(如监测 accuracy)。
2. factor(float)- 学习率调整倍数(等同于其它方法的 gamma)，即学习率更新为 lr = lr * factor
3. patience(int)- 忍受该指标多少个 step 不变化，当忍无可忍时，调整学习率。
4. verbose(bool)- 是否打印学习率信息， print(‘Epoch {:5d}: reducing learning rate of group {} to {:.4e}.’.format(epoch, i, new_lr))
threshold_mode(str)- 选择判断指标是否达最优的模式，有两种模式， rel 和 abs

### LambdaLR

为不同参数组设定不同学习率调整策略。

    torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

1. lr_lambda(function or list)- 一个计算学习率调整倍数的函数，输入通常为 step，当有多个参数组时，设为 list。







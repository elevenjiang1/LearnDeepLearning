# Learn Deep Learning

> 2021.12.17:
>
> ​	仍然感觉有非常多神经网络的操作是不清楚其内部细节的,为此这里面进行内容的详细测试,希望能明确他们的作用等的工作

# 一 网络模型

## 1. 网络层构建

### 1.1 BatchNormal

> 一直不知道eval和train,以及BatchNormal到底是怎么操作的,为此进行验证

**困惑点:**

1. eval和train的区别是是否对数据进行归一化吗?感觉如果直接关掉数据归一化,网络的输出会大改啊
2. 在很多网络训练过程中,神经网络经常会直接输出为0,这个时候必须要加入BatchNormal,这样子网络才能训的动,这个又是为什么



**回答:**

​	最后发现,其实在trian() mode时,会对送入的数据进行减去送入数据均值除以送入数据方差的操作;但是在eval() mode中,其实是减去running_mean除以running_var的操作,这两个参数是固定的,不会因为送入数据的不同而不同.

​	这个数据可能是通过trian的过程中一点点学习从而有的,或者其他的什么方式,这个就需要后续有心情再好好看了...或者扒原文去好好看看之类的





## 2. 损失函数构建







# 二 AutoGrad







# 三 评价指标

## 1. 分类评价指标

对于分类任务,曾经以为就就一个简单的error_rate就可以解决问题的,但是马上发现了其中的巨大问题,需要再去研究.主要就是AP,PR曲线,AUC曲线等等的内容

在DexNet的分类任务中,因为正负样本分布不均衡,有1:4的水平,从而导致了网络就算全部判断为负样本也有21%的error_rate,算是一个不错的水平了;

为此需要专门学一下PR,AUC等的内容

主要代码在Evaluate/learn_prauc.py中
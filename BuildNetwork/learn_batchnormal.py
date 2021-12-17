"""
Reference:https://blog.csdn.net/qq_39208832/article/details/117930625#t8
Check List:
1. Difference between network.train and eval mode
"""
import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
Abs_Path=os.path.dirname(os.path.abspath(__file__))


class example(nn.Module):
    def __init__(self):
        super(example, self).__init__()
        self.fc1 = nn.Linear(3, 3,bias=False)
        self.bn = nn.BatchNorm1d(num_features=3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        return x


def check_train_eval():
    """
    **************************1: net parameter**************************
    fc1.weight : Parameter containing:
    tensor([[-0.0627,  0.4806,  0.0272],
            [ 0.5297, -0.5754, -0.1497],
            [-0.3602,  0.0179, -0.5146]], device='cuda:0', requires_grad=True)
    bn.weight : Parameter containing:
    tensor([1., 1., 1.], device='cuda:0', requires_grad=True)
    bn.bias : Parameter containing:
    tensor([0., 0., 0.], device='cuda:0', requires_grad=True)
    1.1: BatchNormal running mean and var:
    tensor([0., 0., 0.], device='cuda:0')
    tensor([1., 1., 1.], device='cuda:0')
    1.2: fc output:
    tensor([[ 0.9801, -1.0703, -1.8683],
            [ 2.3154, -1.6566, -4.4391]], device='cuda:0', grad_fn=<MmBackward>)
    ******************2: Train mode:******************
    train mode output:
    tensor([[-1.0000,  0.9999,  1.0000],
            [ 1.0000, -0.9999, -1.0000]], device='cuda:0',
        grad_fn=<NativeBatchNormBackward>)
    ******************3: Eval mode:******************
    eval mode output:
    tensor([[ 0.8198, -0.9752, -1.3999],
            [ 2.1624, -1.5874, -3.7175]], device='cuda:0',
        grad_fn=<NativeBatchNormBackward>)
    ******************4: Different input:******************
    4.1: net running mean and var after datas
    tensor([ 0.1648, -0.1363, -0.3154], device='cuda:0')
    tensor([0.9891, 0.9172, 1.2305], device='cuda:0')
    4.2: new running mean and var after new datas
    tensor([ 0.1648, -0.1363, -0.3154], device='cuda:0')
    tensor([0.9891, 0.9172, 1.2305], device='cuda:0')
    """
    #1: Prepare data
    datas = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float)
    datas = datas.cuda()
    net = example().cuda()
    
    #2: Fix model parameter
    # save_state={'model_state_dict':net.state_dict()}
    # torch.save(save_state, os.path.join(Abs_Path,"Batchnormal.pth"))
    net.load_state_dict(torch.load(os.path.join(Abs_Path,"Batchnormal.pth"))['model_state_dict'])

    #3: Show all parameter
    print("**************************1: net parameter**************************")
    net.eval()
    for name,parameters in net.named_parameters():
        print(name,":",parameters)

    print("1.1: BatchNormal running mean and var:")
    print(net.bn.running_mean)#0
    print(net.bn.running_var)#1

    print("1.2: fc output:")
    fc_output=net.fc1(datas)
    print(fc_output)

    print("******************2: Train mode:******************")
    net.train()
    out = net(datas)
    print("train mode output:")
    print(out)


    print("******************3: Eval mode:******************")
    net.eval()
    out = net(datas)
    print("eval mode output:")
    print(out)


    print("******************4: Different input:******************")
    print("4.1: net running mean and var after datas")
    print(net.bn.running_mean)
    print(net.bn.running_var)

    datas = torch.tensor([[0,0,0], [0,0,0]], dtype=torch.float).cuda()
    out = net(datas)
    print("4.2: new running mean and var after new datas")
    print(net.bn.running_mean)#same in different input
    print(net.bn.running_var)

def sim_forward():
    """
    ******************1: fc mode output:******************
    fc_output:
    [[ 0.9801 -1.0702 -1.8682]
    [ 2.3154 -1.6564 -4.4389]]
    ******************2: train() mode output:******************
    mean:
    [ 1.64775 -1.3633  -3.15355]
    var:
    [0.44575652 0.08590761 1.65212462]
    out is:
    [[-0.99998878  0.9999418   0.99999697]
    [ 0.99998878 -0.9999418  -0.99999697]]
    ******************3: eval() mode output:******************
    eval mode output is:
    [[ 0.8197759  -0.97513778 -1.39982243]
    [ 2.16240653 -1.58722234 -3.71726417]]
    """
    #1: Get fc output    
    print("******************1: fc mode output:******************")
    input_data=np.array([[1,2,3],[4,5,6]],dtype=np.float)
    #fc:y=xA^T+b
    fc_layer=np.array([[-0.0627,  0.4806,  0.0272],
                       [ 0.5297, -0.5754, -0.1497],
                       [-0.3602,  0.0179, -0.5146]]).T
    
    fc_output=input_data@fc_layer
    print("fc_output:")
    print(fc_output)
    
    
    #2: In train() mode batchnormal output
    print("******************2: train() mode output:******************")
    mean=np.mean(fc_output,axis=0)
    print("mean:")
    print(mean)
    var=np.var(fc_output,axis=0)
    print("var:")
    print(var)
    
    out=(fc_output-mean)/(np.sqrt(var+1e-05))
    print("out is:")
    print(out)

    #3: In eval() mode batchnormal output
    print("******************3: eval() mode output:******************")
    running_mean=np.array([ 0.1648, -0.1363, -0.3154],dtype=np.float)
    running_var=np.array([0.9891, 0.9172, 1.2305],dtype=np.float)
    out=(fc_output-running_mean)/(np.sqrt(running_var+1e-05))
    print("eval mode output is:")
    print(out)



if __name__ == '__main__':
    # check_train_eval()
    sim_forward()

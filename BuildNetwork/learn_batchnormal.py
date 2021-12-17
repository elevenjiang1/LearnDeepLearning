"""
Reference:https://blog.csdn.net/qq_39208832/article/details/117930625#t8
Check List:
1. Difference between network.train and eval
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
        self.fc1 = nn.Linear(3, 3)
        self.bn = nn.BatchNorm1d(num_features=3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        return x


def check_train_eval():
    datas = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float)
    datas = datas.cuda()
    net = example().cuda()
    
    #Fix model parameter
    # save_state={'model_state_dict':net.state_dict()}
    # torch.save(save_state, os.path.join(Abs_Path,"Batchnormal.pth"))
    net.load_state_dict(torch.load(os.path.join(Abs_Path,"Batchnormal.pth"))['model_state_dict'])

    print("net parameter")
    for name,parameters in net.named_parameters():
        print(name,":",parameters)


    print("In train mode:")
    net.train()
    out = net(datas)
    print(out)


    print("In eval mode:")
    net.eval()
    out = net(datas)
    print(out)




if __name__ == '__main__':
    check_train_eval()

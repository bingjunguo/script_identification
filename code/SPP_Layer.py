#coding=utf-8

import math
import torch
import torch.nn.functional as F


class SPPLayer(torch.nn.Module):
    
    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type
        

    def forward(self, x):
        # num:样本数量 c:通道数 h:高 w:宽
        # num: the number of samples
        # c: the number of channels
        # h: height
        # w: width
        num, c, h, w = x.size() 
#         print(x.size())
        for i in range(self.num_levels):
            level = i+1
            kernel_size = (int(math.ceil(h / level)), int(math.ceil(w / level)))
            stride = (int(math.ceil(h / level)), int(math.ceil(w / level)))
            pooling = (int(math.floor((kernel_size[0]*level-h+1)/2)), int(math.floor((kernel_size[1]*level-w+1)/2)))
            
            # update input data with padding
            zero_pad = torch.nn.ZeroPad2d((pooling[1],pooling[1],pooling[0],pooling[0]))
            
            x_new = zero_pad(x)
            
            # update kernel and stride
            h_new = 2*pooling[0] + h
            w_new = 2*pooling[1] + w
            
            kernel_size = (int(math.ceil(h_new / level)), int(math.ceil(w_new / level)))
            stride = (int(math.floor(h_new / level)), int(math.floor(w_new / level)))
            
            # 选择池化方式 
            if self.pool_type == 'max_pool':
                try:
                    tensor = F.max_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
                except Exception as e:
                    print(str(e))
                    print(x.size())
                    print(level)
            else:
                tensor = F.avg_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
              
            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten

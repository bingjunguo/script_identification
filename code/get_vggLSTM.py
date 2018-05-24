#coding=utf-8

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

class VggLSTM(nn.Module):
    def __init__(self, model_type, model_origin_param, input_size, hidden_size, num_layers, output_size, bidirectional, dropout=0.5, batch_first=True):
        super(VggLSTM, self).__init__()

        if model_type == 'vgg16LSTM':
           vgg = models.vgg16(pretrained=False)
        elif model_type == 'vgg16bnLSTM':
           vgg = models.vgg16_bn(pretrained=False)
        elif model_type == 'vgg19bnLSTM':
           vgg = models.vgg19_bn(pretrained=False)

        vgg.load_state_dict(torch.load(model_origin_param))
        for param in vgg.parameters():
            param.requires_grad = False

        # 提取特征
        self.features = vgg.features # 512*7*7 

        # LSTM
        self.rnn = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = batch_first,
            bidirectional = bidirectional,
            dropout=dropout
        )

        # 如果是双向LSTM，rnn得到的节点数需要乘以2
        if bidirectional:
            self.classifier = torch.nn.Sequential(torch.nn.Linear(hidden_size*2, 1024),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(p=dropout),
                                            torch.nn.Linear(1024, 1024),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(p=dropout),
                                            torch.nn.Linear(1024, output_size))
        else:
            self.classifier = torch.nn.Sequential(torch.nn.Linear(hidden_size, 1024),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(p=dropout),
                                            torch.nn.Linear(1024, 1024),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(p=dropout),
                                            torch.nn.Linear(1024, output_size))

    def forward(self, x):
        output = self.features(x) # num*512*a*b
        n, c, h, w = output.size()
        output, (h_n, h_c) = self.rnn(output.permute(0,2,3,1).contiguous().view(n, h*w, c), None)
        # output, (h_n, h_c) = self.rnn(output.view(-1,512,49), None)
        output = self.classifier(output[:, -1, :])
        
        return output
    

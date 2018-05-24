#coding=utf-8

import torch
import torchvision.models as models
from collections import namedtuple
from SPP_Layer import SPPLayer

def ResnetBaseModel(model_type, model_origin_param, output_size):
    if model_type == 'resnet50':
        model = models.resnet50(pretrained=False)
    elif model_type == 'resnet152':
        model = models.resnet152(pretrained=False)
    model.load_state_dict(torch.load(model_origin_param))

    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Linear(2048, output_size)
        
    return model


def ResnetSPPModel(model_type, model_origin_param, dropout, SPP_layers, output_size):
    if model_type == 'resnet50':
        model = models.resnet50(pretrained=False)
    elif model_type == 'resnet152':
        model = models.resnet152(pretrained=False)
    model.load_state_dict(torch.load(model_origin_param))

    for param in model.parameters():
        param.requires_grad = False

    model.features.add_module('SPP', SPPLayer(SPP_layers))
    in_features = sum([i**2 for i in range(1,SPP_layers+1)])
    model.classifier = torch.nn.Sequential(torch.nn.Linear(in_features, 4096),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p=dropout),
                                        torch.nn.Linear(4096, 4096),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p=dropout),
                                        torch.nn.Linear(4096, output_size))
    return model


class Resnet50Model(torch.nn.Module):
    def __init__(self,  model_origin_param, dropout, feature_layers, in_features, output_size, requires_grad=False):
        super(Resnet50Model, self).__init__()
        self.feature_layers = feature_layers

        resnet = models.resnet50(pretrained=False)
        resnet.load_state_dict(torch.load(model_origin_param))
        if not requires_grad:
            for param in resnet.parameters():
                param.requires_grad = False

        self.layer0 = torch.nn.Sequential()
        self.layer0.add_module('conv1', resnet.conv1)
        self.layer0.add_module('bn1', resnet.bn1)
        self.layer0.add_module('relu', resnet.relu)
        self.layer0.add_module('maxpool', resnet.maxpool)
        
        # self.conv1 = resnet.conv1
        # self.bn1 = resnet.bn1
        # self.relu = resnet.relu
        # self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.classifier = torch.nn.Sequential(torch.nn.Linear(in_features, 4096),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(p=dropout),
                                            torch.nn.Linear(4096, 4096),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(p=dropout),
                                            torch.nn.Linear(4096, output_size))
        
    def forward(self, X):
        n,c,h,w = X.size()
        y0 = self.layer0(X)
        # y0 = self.conv1(X)
        # y0 = self.bn1(y0)
        # y0 = self.relu(y0)
        # y0 = self.maxpool(y0)
        # print('y0 size:{}'.format(y0.size()))

        y1 = self.layer1(y0)
        # print('y1 size:{}'.format(y1.size()))

        y2 = self.layer2(y1)
        # print('y2 size:{}'.format(y2.size()))

        y3 = self.layer3(y2)
        # print('y3 size:{}'.format(y3.size()))

        y4 = self.layer4(y3)
        # print('y4 size:{}'.format(y4.size()))

        resnet_outputs = namedtuple("ResnetOutputs", ['y0', 'y1', 'y2', 'y3', 'y4'])
        all_features = resnet_outputs(y0, y1, y2, y3, y4)

        
        for i in range(len(self.feature_layers)):
            if i == 0:
                index = self.feature_layers[i]
                classifier_features = all_features[index].view(n,-1)  
            else:
                index = self.feature_layers[i]
                classifier_features = torch.cat((classifier_features, all_features[index].view(n,-1)),1)
                
        # print(classifier_features.size())
        output = self.classifier(classifier_features)
        return output

def ResnetSkipModel(model_type, model_origin_param, dropout, feature_layers, in_features, output_size, requires_grad=False):
    if model_type == 'resnet50':
        model = Resnet50Model(model_origin_param, dropout, feature_layers, in_features, output_size, requires_grad)
    elif model_type == 'resnet152':
        pass
    return model
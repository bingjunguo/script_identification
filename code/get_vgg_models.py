#coding=utf-8

import torch

from torchvision import models
from collections import namedtuple
from SPP_Layer import SPPLayer


def VggBaseModel(model_type, model_origin_param, dropout, output_size):
    
    if model_type == 'vgg13':
        model = models.vgg13(pretrained=False) 
    elif model_type == 'vgg16':
        model = models.vgg16(pretrained=False)
    elif model_type == 'vgg16bn':
        model = models.vgg16_bn(pretrained=False)
    elif model_type == 'vgg19bn':
        model = models.vgg19_bn(pretrained=False) 

    model.load_state_dict(torch.load(model_origin_param))

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 重新定义分类器，该分类器各层的requires_grad默认为True，所以参数不会冻结，而是会更新
    model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(p=dropout),
                                    torch.nn.Linear(4096, 4096),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(p=dropout),
                                    torch.nn.Linear(4096, output_size))
    return model



def VggSPPModel(model_type, model_origin_param, dropout, SPP_layers, output_size):
    
    if model_type == 'vgg13':
        model = models.vgg13(pretrained=False) 
    elif model_type == 'vgg16':
        model = models.vgg16(pretrained=False)
    elif model_type == 'vgg16bn':
        model = models.vgg16_bn(pretrained=False)
    elif model_type == 'vgg19bn':
        model = models.vgg19_bn(pretrained=False) 

    model.load_state_dict(torch.load(model_origin_param))

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 重新定义分类器，该分类器各层的requires_grad默认为True，所以参数不会冻结，而是会更新   
    model.features.add_module('SPP', SPPLayer(SPP_layers))
    in_features = sum([i**2 for i in range(1,SPP_layers+1)])*512
    model.classifier = torch.nn.Sequential(torch.nn.Linear(in_features, 4096),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p=dropout),
                                        torch.nn.Linear(4096, 4096),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p=dropout),
                                        torch.nn.Linear(4096, output_size))
    return model

# 将多个卷积层后的特征连接才一起
class VggSkipModel(torch.nn.Module):
    def __init__(self, model_type, model_origin_param, dropout, feature_layers, in_features, output_size, requires_grad=False):
        super(VggSkipModel, self).__init__()
        self.dropout = dropout
        self.feature_layers = feature_layers

        if model_type == 'vgg13':
            vgg = models.vgg13(pretrained=False) 
            features_index = [
                (0,5),
                (5,10),
                (10,15),
                (15,20),
                (20,25)
            ]
        elif model_type == 'vgg16':
            vgg = models.vgg16(pretrained=False)
            features_index = [
                (0,5),
                (5,10),
                (10,17),
                (17,24),
                (24,31)
            ]
        elif model_type == 'vgg16bn':
            vgg = models.vgg16_bn(pretrained=False)
            features_index = [
                (0,7),
                (7,14),
                (14,24),
                (24,34),
                (34,44)
            ]
        elif model_type == 'vgg19bn':
            vgg = models.vgg19_bn(pretrained=False) 
            features_index = [
                (0,7),
                (7,14),
                (14,27),
                (27,40),
                (40,53)
            ]
        
        vgg.load_state_dict(torch.load(model_origin_param))
        if not requires_grad:
            for param in vgg.parameters():
                param.requires_grad = False

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        

        for x in range(features_index[0][0],features_index[0][1]):
            self.slice1.add_module(str(x), vgg.features[x])
        for x in range(features_index[1][0],features_index[1][1]):
            self.slice2.add_module(str(x), vgg.features[x])
        for x in range(features_index[2][0],features_index[2][1]):
            self.slice3.add_module(str(x), vgg.features[x])
        for x in range(features_index[3][0],features_index[3][1]):
            self.slice4.add_module(str(x), vgg.features[x])
        for x in range(features_index[4][0],features_index[4][1]):
            self.slice5.add_module(str(x), vgg.features[x])
        
        in_features = in_features
        self.classifier = torch.nn.Sequential(torch.nn.Linear(in_features, 4096),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(p=dropout),
                                            torch.nn.Linear(4096, 4096),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(p=dropout),
                                            torch.nn.Linear(4096, output_size))
        
        self.spp = SPPLayer(4)

    def forward(self, X):
        n,c,h,w = X.size()
        h = self.slice1(X)
        h_relu1_2 = h
        # print('h_relu1_2 size:{}'.format(h_relu1_2.size()))
        
        h = self.slice2(h)
        h_relu2_2 = h
        # print('h_relu2_2 size:{}'.format(h_relu2_2.size()))
        
        h = self.slice3(h)
        h_relu3_3 = h
        # print('h_relu3_3 size:{}'.format(h_relu3_3.size()))
        
        h = self.slice4(h)
        h_relu4_3 = h
        # print('h_relu4_3 size:{}'.format(h_relu4_3.size()))
        
        h = self.slice5(h)
        h_relu5_3 = h
        # print('h_relu5_3 size:{}'.format(h_relu5_3.size()))
        
        vgg_outputs = namedtuple("VggSkipOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        all_features = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)



        for i in range(len(self.feature_layers)):
            if i == 0:
                index = self.feature_layers[i]
                classifier_features = all_features[index].view(n,-1)  
            else:
                index = self.feature_layers[i]
                classifier_features = torch.cat((classifier_features, all_features[index].view(n,-1)),1)
        output = self.classifier(classifier_features)

        return output

#  将每个卷积层后的特征使用SPP层转换成一维特征向量后，将多个特征拼接在一起。
class VggSkipSPPModel(torch.nn.Module):
    def __init__(self, model_type, model_origin_param, dropout, feature_layers, in_features, output_size, requires_grad=False):
        super(VggSkipModel, self).__init__()
        self.dropout = dropout
        self.feature_layers = feature_layers

        if model_type == 'vgg13':
            vgg = models.vgg13(pretrained=False) 
            features_index = [
                (0,5),
                (5,10),
                (10,15),
                (15,20),
                (20,25)
            ]
        elif model_type == 'vgg16':
            vgg = models.vgg16(pretrained=False)
            features_index = [
                (0,5),
                (5,10),
                (10,17),
                (17,24),
                (24,31)
            ]
        elif model_type == 'vgg16bn':
            vgg = models.vgg16_bn(pretrained=False)
            features_index = [
                (0,7),
                (7,14),
                (14,24),
                (24,34),
                (34,44)
            ]
        elif model_type == 'vgg19bn':
            vgg = models.vgg19_bn(pretrained=False) 
            features_index = [
                (0,7),
                (7,14),
                (14,27),
                (27,40),
                (40,53)
            ]
        
        vgg.load_state_dict(torch.load(model_origin_param))
        if not requires_grad:
            for param in vgg.parameters():
                param.requires_grad = False

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        

        for x in range(features_index[0][0],features_index[0][1]):
            self.slice1.add_module(str(x), vgg.features[x])
        for x in range(features_index[1][0],features_index[1][1]):
            self.slice2.add_module(str(x), vgg.features[x])
        for x in range(features_index[2][0],features_index[2][1]):
            self.slice3.add_module(str(x), vgg.features[x])
        for x in range(features_index[3][0],features_index[3][1]):
            self.slice4.add_module(str(x), vgg.features[x])
        for x in range(features_index[4][0],features_index[4][1]):
            self.slice5.add_module(str(x), vgg.features[x])
        
        in_features = in_features
        self.classifier = torch.nn.Sequential(torch.nn.Linear(in_features, 4096),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(p=dropout),
                                            torch.nn.Linear(4096, 4096),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(p=dropout),
                                            torch.nn.Linear(4096, output_size))
        
        self.spp = SPPLayer(4)

    def forward(self, X):
        n,c,h,w = X.size()
        h = self.slice1(X)
        # h_relu1_2 = h
        h1 = self.spp(h)
        # print('h_relu1_2 size:{}'.format(h_relu1_2.size()))
        
        h = self.slice2(h)
        # h_relu2_2 = h
        h2 = self.spp(h)
        # print('h_relu2_2 size:{}'.format(h_relu2_2.size()))
        
        h = self.slice3(h)
        # h_relu3_3 = h
        h3 = self.spp(h)
        # print('h_relu3_3 size:{}'.format(h_relu3_3.size()))
        
        h = self.slice4(h)
        # h_relu4_3 = h
        h4 = self.spp(h)
        # print('h_relu4_3 size:{}'.format(h_relu4_3.size()))
        
        h = self.slice5(h)
        # h_relu5_3 = h
        h5 = self.spp(h)
        # print('h_relu5_3 size:{}'.format(h_relu5_3.size()))
        
        # vgg_outputs = namedtuple("VggSkipOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        # all_features = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        vgg_spp_outputs = namedtuple('VggSkipSPPOutputs', ['h1', 'h2', 'h3', 'h4', 'h5'])
        all_spp_features = vgg_spp_outputs(h1, h2, h3, h4, h5)

        for i in range(len(self.feature_layers)):
            if i ==0:
                index = self.feature_layers[i]
                classifier_features = all_spp_features[index]
            else:
                index = self.feature_layers[i]
                classifier_features = torch.cat((classifier_features, all_spp_features[index]), 1)

        # for i in range(len(self.feature_layers)):
        #     if i == 0:
        #         index = self.feature_layers[i]
        #         classifier_features = all_features[index].view(n,-1)  
        #     else:
        #         index = self.feature_layers[i]
        #         classifier_features = torch.cat((classifier_features, all_features[index].view(n,-1)),1)
        output = self.classifier(classifier_features)

        return output
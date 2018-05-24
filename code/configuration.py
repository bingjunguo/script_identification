#coding=utf-8

import math
import json

'''
=============================================
输出类别数量
============================================
'''
output_size = 13


'''
=============================================
数据路径
============================================
'''
# data_path = '../data/SIW-10-augmentation/'
# data_path = '../data/SIW-13-augmentation/'
# data_path = '../data/SIW-13-aug-equal/'
data_path = '../data/SIW-13-cut-train/'
# data_path = '../data/SIW-13-train-test/'
# data_path = '../data/SIW-13-latin/'


'''
=============================================
模型类型
============================================
'''

# model_type = 'vgg16bnLSTM'    
# model_type = 'vgg16bn'  
model_type = 'vgg16'    
# model_type = 'vgg13'
# model_type = 'vgg19bn'
# model_type = 'resnet50'
# model_type = 'resnet152'
# model_type = 'singleLSTM'
# model_type = 'BiLSTM'

'''
=============================================
LSTM网络参数
============================================
'''
input_size = 512
hidden_size = 512
num_layers = 2
bidirectional = True

'''
=============================================
输入输出参数
============================================
'''

# resize = [32, 96]
# resize = [50,120]
# resize = [40,110]
# resize = [160, 480]
resize = [224, 224*3]

# feature_type = 'base'   # 直接使用原模型
# feature_type = 'SPP'    # 对最后一层采用SPP得到特征
# feature_type = 'skip'   # 使用不同层次的特征
feature_type = 'skip_spp'   # 使用不同层次的特征

# SPP
SPP_layers = 4


# Skip 选择需要进行特征拼接的层次（VGG共5层）
# feature_layers = [2,3,4]
feature_layers = [1,2,3,4]
# feature_layers = [4]

# Skip分类器输入特征大小
in_features = 0
display = True

if model_type[:3] == 'vgg' and feature_type == 'skip_spp':
    channels = [64,128,256,512,512]     # vgg
    layers_num = []     # 记录每一层的数量
    for layer in range(len(channels)):
        out_features = sum([i**2 for i in range(1,SPP_layers+1)])*channels[layer]
        layers_num.append(out_features)
        if display:
            print('layer:{} num:{}'.format(layer,out_features))
    for layer in feature_layers:
        in_features += layers_num[layer]

elif model_type[:3] == 'vgg' and feature_type == 'skip':
    channels = [64,128,256,512,512]     # vgg
    x = math.floor(resize[0])
    y = math.floor(resize[1])
    layers_num = []     # 记录每一层的数量
    for layer in range(len(channels)):
        x = math.floor(x/2)
        y = math.floor(y/2)
        layers_num.append(channels[layer]*x*y)
        if display:
            print('layer:{} x:{} y:{} num:{}'.format(layer,x,y,channels[layer]*x*y))
    for layer in feature_layers:
        in_features += layers_num[layer]
elif model_type[:3] == 'res' and feature_type == 'skip':
    channels = [64,256,512,1024,2048]   # resnet
    layers_num = []     # 记录每一层的数量
    for layer in range(len(channels)):
        if layer <= 1:
            x = math.ceil(resize[0]/4)
            y = math.ceil(resize[1]/4)
            layers_num.append(channels[layer]*x*y)
        else:
            x = math.ceil(x/2)
            y = math.ceil(y/2)
            layers_num.append(channels[layer]*x*y)
        if display:
            print('layer:{} x:{} y:{} num:{}'.format(layer,x,y,channels[layer]*x*y))
    for layer in feature_layers:
        in_features += layers_num[layer]

'''
=============================================
网络参数
============================================
'''
# 决定是否导入已经训练好的模型参数
# model_trained = True
model_trained = False

# 优化器选择
# optim_type = 'SGD'
optim_type = 'Adam'
# optim_type = 'RMSprop'

batch_size = 32
dropout = 0.5
weight_decay = 0
lr = 1e-4
step_size = 20
EPOCH = 80


'''
=============================================
param to key-value
=============================================
'''


param_loader = {
    'name': 'param_loader',

    # 数据路径
    'data_path': data_path,

    # batch大小
    'batch_size': batch_size,

    # 图片transform resize的大小
    'resize': resize,

    # loader的并行线程数
    'nThreads':
        2,
        # 3,
        # 4, 
}

param_model = {
    'name': 'param_model',

    # 输入图像大小
    'resize': resize,

    # 模型类型
    'model_type': model_type,

    # 对应原始模型参数路径
    'model_origin_param':{
        'vgg16LSTM': '../data/models/vgg16_origin.pth',
        'vgg16bnLSTM': '../data/models/vgg16bn_origin.pth',
        'vgg19bnLSTM': '../data/models/vgg19bn_origin.pth',
        'vgg13': '../data/models/vgg13_origin.pth',
        'vgg16': '../data/models/vgg16_origin.pth',
        'vgg16bn': '../data/models/vgg16bn_origin.pth',
        'vgg19bn': '../data/models/vgg19bn_origin.pth',
        'resnet50': '../data/models/resnet50_origin.pth',
        'resnet152': '../data/models/resnet152_origin.pth',
        'singleLSTM': '../data/models/singleLSTM_origin.pth',
        'BiLSTM': '../data/models/BiLSTM_origin.pth',
    },

    # 是否有已经训练好的模型参数
    'model_trained': model_trained,

    'model_trained_param':{
        'vgg13': '',
        'vgg16': '',
        'vgg19bn': '',
        'resne50t':'',
        'resnet152': '',
        'singleLSTM':'',
        'BiLSTM': '',
    },
    # dropout
    'dropout': dropout,

    # LSTM网络参数
    'input_size': input_size,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'output_size': output_size,
    'bidirectional': bidirectional,

    # 模型特征方法
    'feature_type': feature_type, 

    'SPP_layers': SPP_layers,

    # 选择的特征层次
    'feature_layers': feature_layers,

    # 第一个全连接层的输入特征数
    # 需要手动修改
    'in_features': in_features,

}

param_optim = {
    'name': 'param_optim',
    
    # 优化器类型
    'optim_type': optim_type,

    # 权重衰减
    'weight_decay': weight_decay,

    # 学习率
    'lr': lr,
    
    # 惯性
    'momentum': 0.9

}

param_lr_scheduler = {
    'name': 'param_lr_scheduler',
    
    # 优化学习率步长
    'step_size': step_size,
}

param_run = {
    'name': 'param_run',
    
    # 迭代次数
    'EPOCH': EPOCH,
}

param_log = {
    'name': 'param_log',
    
    # 日志文件名
    'log_file': '',
}

def get_params(debug=False):
    acc = 0.88538 # 需要手动修改
    param_model['model_trained_param'][model_type] = '../data/models/{}_{}_{}.pkl'.format(model_type, feature_type, acc)
    log_file = './logs/{}/{}_{}.txt'.format(model_type, model_type, feature_type)
    param_log['log_file'] = log_file
    if debug:
        pass
        # for p in [param_loader, param_model, param_optim, param_lr_scheduler, param_run, param_log]:
        #     print('\n{}:'.format(p['name']))
        #     p = json.dumps(p, indent=1)
        #     print(p)
        #     print('\n+++++++++++++++++++++++++++++++++++++++')
    else:
        with open(log_file, 'a+') as f:
            print('\n\n\n\n\n\n==================================================================\n', file=f)
            print('Model_type:{} feature_type:{} feature_layers:{} in_features:{}'.format(model_type, feature_type, feature_layers, in_features), file=f)
            print('Model_type:{} feature_type:{} feature_layers:{} in_features:{}'.format(model_type, feature_type, feature_layers, in_features))
    
            for p in [param_loader, param_model, param_optim, param_lr_scheduler, param_run, param_log]:
                print('\n{}:'.format(p['name']), file=f)
                print('\n{}:'.format(p['name']))
                p = json.dumps(p, indent=1)
                print('{}'.format(p), file=f)
                print('{}'.format(p))
                print('+++++++++++++++++++++++++++++++++++++++\n', file=f)
                print('+++++++++++++++++++++++++++++++++++++++\n')
    return param_loader, param_model, param_optim, param_lr_scheduler, param_run, param_log


if __name__ == '__main__':
    print('Model_type:{} feature_type:{} feature_layers:{} in_features:{}'.format(model_type, feature_type, feature_layers, in_features))
    debug = True
    a,b,c,d,e,f = get_params(debug)
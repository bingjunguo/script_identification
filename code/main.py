#coding=utf-8

import os
import time

import numpy as np
import sklearn
import torch
from sklearn.model_selection import train_test_split
from torch import optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from choose_model import getModel
from configuration import get_params
from run_model import run

param_loader, param_model, param_optim, param_lr_scheduler, param_run, param_log = get_params()

# dataloader参数
data_path = param_loader['data_path']
batch_size = param_loader['batch_size']
resize = param_loader['resize']
nThreads = param_loader['nThreads']

# model参数
model_type = param_model['model_type']
dropout = param_model['dropout']
model_origin_param = param_model['model_origin_param'][model_type]
feature_type = param_model['feature_type']


# 优化器参数
optim_type = param_optim['optim_type']
weight_decay = param_optim['weight_decay']
lr = param_optim['lr']
momentum = param_optim['momentum']

# 学习率优化器参数
step_size = param_lr_scheduler['step_size']

# 迭代次数
EPOCH = param_run['EPOCH']

# 日志文件名
log_file = param_log['log_file']



def main():
    f = open(log_file, 'a+')
    print('\n\n\n\n\n=======================================================================================', file=f)
    print('\n\n\n\n\n=======================================================================================')
    print(time.strftime("\n-------------------\n%Y-%m-%d %H:%M:%S\n-------------------", time.localtime()), file=f)
    print(time.strftime("\n-------------------\n%Y-%m-%d %H:%M:%S\n-------------------", time.localtime()))
        

    '''
    =============================================================================================================================
    使用ImageFolder导入数据
    =============================================================================================================================
    '''
    if model_type[-4:] == 'LSTM' and model_type[:3] != 'vgg':
        data_fixed_transforms = {   
            'train': transforms.Compose([
                transforms.Resize((resize[0], resize[1])),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((resize[0], resize[1])),
                transforms.Grayscale(1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    else:
        data_fixed_transforms = {   
        'train': transforms.Compose([
            transforms.Resize((resize[0], resize[1])),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((resize[0], resize[1])),
            # transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    print('Datasets:{}'.format(data_path), file=f)
    print('Datasets:{}'.format(data_path))

    image_datasets = {x: datasets.ImageFolder(data_path+x,
                                            data_fixed_transforms[x])
                    for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'test']}
    # dataloaders = {
        # 'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
        #                                         shuffle=True, num_workers=4),
        # 'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=1,
        #                                         shuffle=True, num_workers=4)
        # }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    for class_ in class_names:
        print('{}:{}'.format(class_, len(os.listdir(data_path+'train/'+class_))), file=f)
        print('{}:{}'.format(class_, len(os.listdir(data_path+'train/'+class_))))
    print(dataset_sizes, file=f)
    print(dataset_sizes)

    '''
    =============================================================================================================================
    构建网络
    =============================================================================================================================
    '''

    # 是否使用GPU
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False

    print("Building {}  ...".format(model_origin_param), file=f)
    print("Building {}  ...".format(model_origin_param))
    
    model = getModel(param_model, f)
    print('{}'.format(model), file=f)
    print('{}'.format(model))

    if use_cuda:
        model = model.cuda()


    '''
    =============================================================================================================================
    构建损失函数、优化器
    =============================================================================================================================
    '''
    criterion = torch.nn.CrossEntropyLoss()

    # 优化器
    if optim_type == 'SGD':
        optimizer_ft = optim.SGD(model.classifier.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim_type == 'Adam':
        optimizer_ft = optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type == 'RMSprop':
        optimizer_ft = optim.RMSprop(model.classifier.parameters(), lr=lr, alpha=0.9, weight_decay=weight_decay)
    print('optimizer:{}'.format(optimizer_ft), file=f)
    print('optimizer:{}'.format(optimizer_ft))

    # 动态改变学习率
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=0.1)
    print('lr_scheduler:{}'.format(exp_lr_scheduler), file=f)
    print('lr_scheduler:{}'.format(exp_lr_scheduler))

    # 打印出参数
    print("LR={},dropout={},EPOCH={},batch_size={}, weight_decay={}".format(lr, dropout, EPOCH, batch_size, weight_decay), file=f)
    print("LR={},dropout={},EPOCH={},batch_size={}, weight_decay={}".format(lr, dropout, EPOCH, batch_size, weight_decay))

    run(model_type, feature_type, model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, use_cuda, f, num_epochs=EPOCH)
    f.close()

if __name__ == '__main__':
    debug = False
    if debug:
        pass
    else:
        start = time.time()
        main()
        end = time.time() - start
        with open(log_file, 'a+') as f:
            print("Total time:{:.0f}m {:.0f}s".format(end//60, end%60), file=f)
            print("Total time:{:.0f}m {:.0f}s".format(end//60, end%60))

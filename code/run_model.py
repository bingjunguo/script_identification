#coding=utf-8

import copy
import time

import torch
from torch.autograd import Variable
from sklearn.metrics import classification_report

from tensorboardX import SummaryWriter

def run(model_type, feature_type, model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, use_cuda, f, num_epochs=25):
    '''
    model: 模型
    criterion: 损失函数
    optimizer: 优化器
    scheduler：学习率优化器
    dataloaders：字典(dict),包含train和test的dataloader
    dataset_sizes：train和test数据集的大小
    use_cuda：是否使用cuda
    f：日志文件描述符
    num_epochs：迭代次数
    '''

    log_dir = './runs/{}_{}'.format(model_type, feature_type)

    # tensorboardX
    writer = SummaryWriter(log_dir)


    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # i表示记录节点的起始位置。
    i = 0
    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1), file=f)
        print('-' * 10, file=f)
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
                print(scheduler.get_lr())
                print(scheduler.get_lr(), file=f)
            else:
                model.eval()
                # model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_acc = 0

            num = 0
            # Iterate over data.
            for data in dataloaders[phase]:
                inputs, labels = data

                if phase == 'train':
                    i += 1

                # wrap them in Variable
                if use_cuda:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_acc += torch.sum(preds == labels.data)
                
                num += labels.data.size(0)
                if phase == 'train' and i%100==0:
                    log_loss = running_loss/num
                    log_acc  = running_acc /num
                    writer.add_scalar('{}/{}_{}_loss'.format(phase, model_type, feature_type), log_loss, i)
                    writer.add_scalar('{}/{}_{}_acc'.format(phase, model_type, feature_type), log_acc, i)
               
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), file=f)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), file=f)
    print('Best val Acc: {:4f}'.format(best_acc), file=f)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    print('Save ../data/models/{}_{}_{}.pkl'.format(model_type, feature_type, best_acc), file=f)
    print('Save ../data/models/{}_{}_{}.pkl'.format(model_type, feature_type, best_acc))
    torch.save(best_model_wts, '../data/models/{}_{}_{}.pkl'.format(model_type, feature_type, best_acc))

    # 评估模型
    for step, (inputs, labels) in enumerate(dataloaders['test']):
        # wrap them in Variable
        if use_cuda:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        model.eval()
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        if step == 0:
            y_true = labels
            y_pred = preds
        else:
            y_true = torch.cat((y_true, labels), -1)
            y_pred = torch.cat((y_pred, preds), -1)
                
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    # 评估类别名称
    langs = ['Arabic', 'Cambodian','Chinese','English','Greek','Hebrew','Japanese','Kannada','Korean','Mongolian','Russian','Thai','Tibetan']
    target_names = []
    target_num = 13
    for i in range(target_num):
        name = str(i+1)+'_'+langs[i]
        target_names.append(name)
    print(classification_report(y_true_np, y_pred_np, target_names=target_names))

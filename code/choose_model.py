#coding=utf-8

import torch
from tensorboardX import SummaryWriter

from get_LSTM import LSTMBaseModel
from get_vggLSTM import VggLSTM
from get_resnet_models import ResnetBaseModel, ResnetSkipModel, ResnetSPPModel
from get_vgg_models import VggBaseModel, VggSkipModel, VggSPPModel, VggSkipSPPModel


def getModel(param_model, f):
    resize = param_model['resize']
    model_type = param_model['model_type']
    feature_type = param_model['feature_type']
    model_origin_param = param_model['model_origin_param'][model_type]
    model_trained = param_model['model_trained']
    model_trained_param = param_model['model_trained_param'][model_type]
    dropout = param_model['dropout']
    SPP_layers = param_model['SPP_layers']
    feature_layers = param_model['feature_layers']
    in_features = param_model['in_features']
    input_size = param_model['input_size']
    hidden_size = param_model['hidden_size']
    num_layers = param_model['num_layers']
    output_size = param_model['output_size']
    bidirectional = param_model['bidirectional']

    # print('Building {}{}'.format(model_type, feature_type), file=f)
    # print('Building {}{}'.format(model_type, feature_type))
    if feature_type == 'base':
        if model_type[:3] == 'vgg' and model_type[-4:] == 'LSTM':
            model = VggLSTM(model_type, model_origin_param, input_size, hidden_size, num_layers, output_size, bidirectional, dropout, True)
        elif model_type[:3] == 'vgg':
            model = VggBaseModel(model_type, model_origin_param, dropout, output_size)
        elif model_type[-4:] == 'LSTM':
            model = LSTMBaseModel(model_type, input_size, hidden_size, num_layers, output_size, dropout)
        elif model[:3] == 'res':
            model = ResnetBaseModel(model_type, model_origin_param, output_size)
     
    elif feature_type == 'SPP':
        if model_type[:3] == 'vgg':
            model = VggSPPModel(model_type, model_origin_param, dropout, SPP_layers, output_size)
        elif model_type[:3] == 'res':
            model = ResnetSPPModel(model_type, model_origin_param, dropout, SPP_layers, output_size)
       
    elif feature_type == 'skip':
        if model_type[:3] == 'vgg':
            model = VggSkipModel(model_type, model_origin_param, dropout, feature_layers, in_features, output_size, False)
        elif model_type[:3] == 'res':
            model = ResnetSkipModel(model_type, model_origin_param, dropout, feature_layers, in_features, output_size, False)
    
    elif feature_type == 'skip_spp':
        if model_type[:3] == 'vgg':
            model = VggSkipSPPModel(model_type, model_origin_param, dropout, feature_layers, in_features, output_size, False)

    if model_trained:
        print('Loading {}'.format(model_trained_param), file=f)
        print('Loading {}'.format(model_trained_param))
        model.load_state_dict(torch.load(model_trained_param))

    # 模型可视化
    h,w = resize[0],resize[1]
    res = torch.autograd.Variable(torch.Tensor(1,3,h,w), requires_grad=True)
    writer = SummaryWriter()
    writer.add_graph(model, res)
    writer.close()
    return model

# script_identification
基于卷积神经网络文字语种识别算法  
the method based on CNN for script identification

这是本科毕业设计的课题，主要使用基于迁移学习卷积神经网络实现对文字语种图像分类。

## 数据集

数据集采用SIW-13，共有13个语种（英文、中文、日文、希腊文、俄文、泰文、阿拉伯文、柬埔寨文、希伯来文、卡纳达文、韩文、蒙古文、藏文）。


## 实验平台

- Python3.6
- Pytorch < 0.4 (Pytorch 0.4做了一些较大的修改，如若想查看本代码效果，建议使用低于0.4版本的Pytorch)

## 文件结构

​
![image.png](https://ask.qcloudimg.com/draft/1215004/i40rvxblnk.png)

```shell
C:.
│  README.md
│
├─code
│  │  main.py：主文件，读取参数和模型，调用run_model.py
│  │  configuration.py：设置各种参数
│  │  run_model.py：运行模型
│  │  choose_model.py：根据参数选择模型
│  │  get_LSTM.py：单纯使用LSTM，在该数据集下表现不佳
│  │  get_vggLSTM.py：vgg提取特征，LSTM训练分类器
│  │  get_resnet_models.py
│  │  get_vgg_models.py
│  │  SPP_Layer.py：单独写的spp层
│  │
│  ├─logs：保存输出的可视化数据。（需要安装tensorboardX库和TensorFlow）
│  └─runs：存储输出的日志，日志信息包括参数配置，模型结构，每次迭代的训练集和测试集的准确度
└─data
    └─models
```

- **data**文件夹中还有一个models文件夹，用于保存训练得到的模型，以及像vgg16这样的预训练模型参数。
- **get_vgg_models.py**中共有4中vgg派生模型：
    - **VggBaseModel**：vgg的原模型
    - **VggSPPModel**：vgg第一个全连接层之前加上spp层，这样可以固定全连接层输入节点数，所以可以接收任意大小的输入图像 
    - **VggSkipModel**：将多个卷积层后的特征连接才一起
    - **VggSkipSPPModel**：将每个卷积层后的特征使用SPP层转换成一维特征向量后，将多个特征拼接在一起。
- **get_resnet_models.py**：同上。但是因为效果不及vgg，所以有的模型没有实现。

## 注意事项

- 优化器建议使用**SGD**，在实验中，Adam和RMSprop效果并没有好很多，而且这两个优化器比较吃显存，是很吃显存！！！


有什么疑问可以email：marsggbo@foxmail.com
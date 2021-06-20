import os
import torch

from torch import optim
from model.pretrained_LeNet import get_pretrained_LeNet_MNIST_classifier
from layer.softmax_with_temperature import SoftmaxWithTemperature
from model.simple_2_layers_linear import get_simple_2_layers_linear
from utils.cross_entropy_loss import cross_entropy_loss
from utils.fit_model import fit_model
from utils.get_MNIST_dataset import get_MNIST_dataset
from utils.get_accuracy import get_accuracy
from utils.get_parameter_number import get_parameter_number

# batch size
bs = 64

# 教师网络
net_T = get_pretrained_LeNet_MNIST_classifier()
net_T.fc.sig7 = SoftmaxWithTemperature(T=1)
print('教师网络参数情况为：', get_parameter_number(net_T))
print('教师网络分类准确率为：', get_accuracy(net_T, False))

# 获取DataLoader
dataset_train = get_MNIST_dataset(is_train=True)
dataset_val = get_MNIST_dataset()
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=bs, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset_val, batch_size=bs * 2, shuffle=True)

# 原始网络结构
# 存在就取现成的
path = './model/original_small'
if os.path.exists(path):
    original_net_S = torch.load(path)
# 否则训练一个
else:
    original_net_S = get_simple_2_layers_linear(T=1)
    # 单独训练这个简单的2层线性relu网络
    fit_model(epochs=10, model=original_net_S, hard_loss_func=cross_entropy_loss,
              opt=optim.Adam(original_net_S.parameters()),
              train_dl=train_loader,
              valid_dl=valid_loader,
              T=1)

    torch.save(original_net_S, path)
print('原始小网络参数情况为：', get_parameter_number(original_net_S))
print('原始小网络分类准确率为：', get_accuracy(original_net_S))


# 特定设置下进行蒸馏
def distill(T, alpha, beta, T_square_make_up):
    config = f'T-{T}_alpha-{alpha}_beta-{beta}_MakeUp-{"T" if T_square_make_up else "F"}'

    print("================================")
    print(f'{config}配置下蒸馏')

    T = T
    net_T.fc.sig7 = SoftmaxWithTemperature(T=T)
    savePath = f'./model/distilled_models/{config}_distilled'
    # 存在就取训练好的
    if os.path.exists(savePath):
        print('已训练过')
        model = torch.load(savePath)
    # 否则训练一个
    else:
        print('未训练过，开始训练……')
        model = get_simple_2_layers_linear(T=T)
        # 单独训练这个简单的2层线性relu网络
        fit_model(epochs=10, model=model, hard_loss_func=cross_entropy_loss, opt=optim.Adam(model.parameters()),
                  train_dl=train_loader,
                  valid_dl=valid_loader, T=T, teacher_model=net_T, soft_loss_func=cross_entropy_loss)

        torch.save(model, savePath)
    print(f'{config}时蒸馏分类准确率为：',
          get_accuracy(model))
    print("================================\n")


count = 0
for T in [2, 4, 6, 10]:
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        beta = round(1 - alpha, 1)
        for makeUp in [True, False]:
            count += 1
            print(count)
            distill(T, alpha, beta, makeUp)

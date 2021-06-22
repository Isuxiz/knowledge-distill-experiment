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
from utils.modified_dataset_collaters import exclude_3_collate, only_contain_7_and_8_collate

# batch size
bs = 64
# epoches
eps = 10

# 教师网络
net_T = get_pretrained_LeNet_MNIST_classifier()
net_T.fc.sig7 = SoftmaxWithTemperature(T=1)
print('教师网络参数情况为：', get_parameter_number(net_T))
print('教师网络分类准确率为：', get_accuracy(net_T, is_student_model=False))

# 获取DataLoader
dataset_train = get_MNIST_dataset(is_train=True)
dataset_val = get_MNIST_dataset()

# 验证集
valid_loader = torch.utils.data.DataLoader(dataset_val, batch_size=bs * 2, shuffle=True)

# 常规训练集
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=bs, shuffle=True)

# 排除3这一类的训练集
train_loader_excluded_3 = torch.utils.data.DataLoader(dataset_train, batch_size=bs, shuffle=True,
                                                      collate_fn=exclude_3_collate)
# 仅包含7和8这两类的训练集
train_loader_only_contain_7_and_8 = torch.utils.data.DataLoader(dataset_train, batch_size=bs, shuffle=True,
                                                                collate_fn=only_contain_7_and_8_collate)

# 所有小网络共享同一份初始权值
if not os.path.exists('./model/uniform_initial_weight'):
    torch.save(get_simple_2_layers_linear(T=1).state_dict(), './model/uniform_initial_weight')


# 无教师的单独训练
def train_simply(train_loader, valid_loader, saveFileSuffix=None, will_print_paras=False):
    config = 'original_small'
    if saveFileSuffix:
        config += '_' + saveFileSuffix
    savePath = f'./model/{config}'

    print("================================")
    print(f'{config}配置下无教师指导训练')

    # 存在就取现成的
    if os.path.exists(savePath):
        if torch.cuda.is_available():
            simple_model = torch.load(savePath)
        else:
            simple_model = torch.load(savePath, map_location=torch.device('cpu'))
    # 否则训练一个
    else:
        simple_model = get_simple_2_layers_linear(T=1)
        # 保证所有小模型的初始权重一致
        simple_model.load_state_dict(torch.load('./model/uniform_initial_weight'))
        # 单独训练这个简单的2层线性relu网络
        fit_model(epochs=eps, model=simple_model, hard_loss_func=cross_entropy_loss,
                  opt=optim.Adam(simple_model.parameters()),
                  train_dl=train_loader,
                  valid_dl=valid_loader,
                  T=1)

        torch.save(simple_model, savePath)
    if will_print_paras:
        print(f'{config}的参数情况为：', get_parameter_number(simple_model))
    print(f'{config}的准确率为：', get_accuracy(simple_model))
    print("================================\n")


# 特定设置下进行蒸馏
def train_with_distillation(T, alpha, beta, T_square_make_up, train_loader, valid_loader, saveFileSuffix=None):
    config = f'T-{T}_alpha-{alpha}_beta-{beta}_MakeUp-{"T" if T_square_make_up else "F"}'
    if saveFileSuffix:
        config += '_' + saveFileSuffix

    print("================================")
    print(f'{config}配置下带蒸馏训练')

    T = T
    net_T.fc.sig7 = SoftmaxWithTemperature(T=T)
    savePath = f'./model/distilled_models/{config}_distilled'
    # 存在就取训练好的
    if os.path.exists(savePath):
        print('已训练过')
        if torch.cuda.is_available():
            model = torch.load(savePath)
        else:
            model = torch.load(savePath, map_location=torch.device('cpu'))
    # 否则训练一个
    else:
        print('未训练过，开始训练……')
        model = get_simple_2_layers_linear(T=T)
        # 保证所有小模型的初始权重一致
        model.load_state_dict(torch.load('./model/uniform_initial_weight'))
        # 教师指导训练这个简单的2层线性relu网络
        fit_model(epochs=eps, model=model, hard_loss_func=cross_entropy_loss, opt=optim.Adam(model.parameters()),
                  train_dl=train_loader,
                  valid_dl=valid_loader, T=T, teacher_model=net_T, soft_loss_func=cross_entropy_loss)

        if not os.path.exists(f'./model/distilled_models/'):
            os.mkdir(f'./model/distilled_models/')
        torch.save(model, savePath)
    print(f'{config}时蒸馏分类准确率为：',
          get_accuracy(model))
    print("================================\n")


# 单独训练一个最普通的网络
train_simply(train_loader=train_loader, valid_loader=valid_loader)

# 单独训练一个训练集中没有类型3的小网络
train_simply(train_loader=train_loader_excluded_3, valid_loader=valid_loader, saveFileSuffix='excluded_3')

# 单独训练一个训练集中只有类型7和8的小网络
train_simply(train_loader=train_loader_only_contain_7_and_8, valid_loader=valid_loader,
             saveFileSuffix='only_contain_7_and_8')

# 教师指导训练，不同配置下
# count = 0
# for T in [2, 4, 6, 10, 20]:
#     for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
#         beta = round(1 - alpha, 1)
#         for makeUp in [True, False]:
#             count += 1
#             print(count)
#             distill(T, alpha, beta, makeUp,
#                     train_loader=train_loader, valid_loader=valid_loader)

# 教师指导训练一个训练集中没有类型3的小网络
# 取配置为T=4，alpha=beta=0.5，有makeup
train_with_distillation(T=4, alpha=0.5, beta=0.5, T_square_make_up=True,
                        train_loader=train_loader_excluded_3, valid_loader=valid_loader,
                        saveFileSuffix='excluded_3')

# 教师指导训练一个训练集中只有类型7和8的小网络
# 取配置为T=4，alpha=beta=0.5，有makeup
train_with_distillation(T=4, alpha=0.5, beta=0.5, T_square_make_up=True,
                        train_loader=train_loader_only_contain_7_and_8, valid_loader=valid_loader,
                        saveFileSuffix='only_contain_7_and_8')

import os.path as path
import torchvision.transforms as transforms

from torchvision.datasets.mnist import MNIST
from utils.get_project_root_path import get_project_root_path


def get_MNIST_dataset(is_train=False):
    # 这里有一步转化，从28*28*1的图像转化成了32*32*1，方便多次卷积
    data_root = path.join(get_project_root_path(), 'data')
    return MNIST(data_root,
                 train=is_train,
                 download=True,
                 transform=transforms.Compose([
                     transforms.Resize((32, 32)),
                     transforms.ToTensor()]))

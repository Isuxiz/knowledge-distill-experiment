import os
import os.path as path
import torch
import numpy as np

from model.mnist_classifier.lenet import LeNet5
from utils.get_MNIST_dataset import get_MNIST_dataset

from utils.get_available_device import get_available_device
from utils.get_project_root_path import get_project_root_path


def get_pretrained_LeNet_MNIST_classifier():
    model = LeNet5().eval()
    model.load_state_dict(
        torch.load(
            path.join(get_project_root_path(), 'model', 'mnist_classifier', 'weights',
                      'lenet_epoch=12_test_acc=0.991.pth')))
    return model.to(get_available_device())


if __name__ == "__main__":
    net = get_pretrained_LeNet_MNIST_classifier()
    dataset = get_MNIST_dataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=True)
    (ims, labs) = next(iter(data_loader))  # get whole dataset
    ims = ims.to(get_available_device())
    labs = labs.numpy()

    predictions = net(ims).cpu().detach().numpy()
    predicted_class = np.argmax(predictions, axis=1)
    print('test acc', np.sum(predicted_class == labs) / 10000)

import numpy as np
import torch

from utils.get_MNIST_dataset import get_MNIST_dataset
from utils.get_available_device import get_available_device


def get_accuracy(model, split=True):
    dataset_val = get_MNIST_dataset()
    data_loader = torch.utils.data.DataLoader(dataset_val, batch_size=10000, shuffle=True)
    (x, y_truth) = next(iter(data_loader))  # get whole dataset
    x = x.to(get_available_device())
    output = model(x)
    y_hat = output[1] if split else output
    predictions = y_hat.cpu().detach().numpy()
    predicted_class = np.argmax(predictions, axis=1)
    return np.sum(predicted_class == y_truth.numpy()) / 10000

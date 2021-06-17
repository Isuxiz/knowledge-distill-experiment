import torch


def get_available_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

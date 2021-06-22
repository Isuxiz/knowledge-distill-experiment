import torch
import numpy as np
from torch.utils.data._utils.collate import default_collate


def exclude_3_collate(batch):
    modified_batch = []
    for item in batch:
        image, label = item
        if label != 3:
            modified_batch.append(item)
    return default_collate(modified_batch)


def only_contain_7_and_8_collate(batch):
    modified_batch = []
    for item in batch:
        image, label = item
        if label == 7 or label == 8:
            modified_batch.append(item)
    return default_collate(modified_batch)

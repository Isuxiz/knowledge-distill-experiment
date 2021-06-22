import numpy as np
import torch

from utils.get_MNIST_dataset import get_MNIST_dataset
from utils.get_available_device import get_available_device


class AccFormatter:
    def __init__(self, result_dict):
        self.res = result_dict
        self.tp_total = np.array([x['tp'] for x in self.res.values()]).sum()
        self.fp_total = np.array([x['fp'] for x in self.res.values()]).sum()
        self.fn_total = np.array([x['fn'] for x in self.res.values()]).sum()

        assert (self.tp_total + (self.fp_total + self.fn_total / 2)) != 10000, 'Label number wrong.'

        self.total_acc = self.tp_total / 10000

    def __str__(self):
        string = '\n\t'
        tp = '\nTP\t'
        fp = '\nFP\t'
        fn = '\nFN\t'
        for k, v in self.res.items():
            string += "{0:<4}\t".format(k)
            tp += "{0:<4}\t".format(v['tp'])
            fp += "{0:<4}\t".format(v['fp'])
            fn += "{0:<4}\t".format(v['fn'])
        return string + tp + fp + fn + f'\nTotal acc: {round(self.total_acc * 100, 2)}%\n'


def get_accuracy(model, is_student_model=True):
    dataset_val = get_MNIST_dataset()
    data_loader = torch.utils.data.DataLoader(dataset_val, batch_size=10000, shuffle=True)
    (x, y_truth) = next(iter(data_loader))  # get whole dataset
    x = x.to(get_available_device())
    output = model(x)
    y_hat = output[1] if is_student_model else output
    predictions = y_hat.cpu().detach().numpy()
    predicted_class = np.argmax(predictions, axis=1)
    y_truth = y_truth.numpy()

    # 多分类问题中没有真负例，因为真负例其实是某个其他类的真正例
    # 对3这一类来说，有以下三个指标：
    # TP(真正例)：分类准确，实际是3，预测为3
    # FP(假正例)：分类不准确，实际是4，误预测为3 (此类同时是对4来说的FN)
    # FN(假负例)：分类不准确，实际是3，误预测为4 (此类同时是对4来说的FP)

    result = dict()
    for i in range(10):
        result.setdefault(i, {'tp': 0, 'fp': 0, 'fn': 0})

    for i in range(len(predicted_class)):
        if predicted_class[i] == y_truth[i]:
            result[y_truth[i]]['tp'] += 1
        else:
            result[y_truth[i]]['fn'] += 1
            result[predicted_class[i]]['fp'] += 1

    return AccFormatter(result)

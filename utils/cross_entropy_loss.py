import torch.nn.functional as F


def cross_entropy_loss(P, Q):
    """
    交叉熵函数，H(P||Q)，reduction为sum模式
    :param P: 真实分布，要求是未one-hot化的
    :param Q: 预测分布，要求已经softmax过但未log过
    :return: 交叉熵
    """
    return F.nll_loss(Q.log(), P, reduction="sum")

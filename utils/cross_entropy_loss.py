import torch.nn.functional as F


def cross_entropy_loss(P, Q):
    """
    交叉熵函数，H(P||Q)，reduction为sum模式
    :param P: 真实分布，如果是1维说明是仅标签，如果是2维说明是教师模型预测出的分布
    :param Q: 预测分布，要求已经softmax过但未log过
    :return: 交叉熵
    """
    dim = P.dim()
    if dim == 1:
        # 使用nll_loss自动把label的tensor进行one-hot编码，然后哈达玛乘后求和取负
        return F.nll_loss(Q.log(), P, reduction="sum")
    elif dim == 2:
        return ((Q.log() * P).sum(-1) * -1).sum()
    else:
        assert True, 'P must be a 1-dimensional label tensor\
 (the dimension represents batch size) \
 or 2-dimensional distribution tensor \
 (dimensions represent batch size and various probability distributions respectively).'

import torch
import numpy as np
from utils.get_available_device import get_available_device


def loss_batch(model, hard_loss_func, x_batch, y_batch, T, optimizer=None,
               teacher_model=None, soft_loss_func=None,
               alpha=0.9, beta=0.1, T_square_make_up=False):
    output = model(x_batch)
    # ground_truth标签yb，shape为[batch_size]，即未one-hot化过
    # soft_y_hat和hard_y_hat均为已经softmax过后的结果，shape均为[batch_size, 10]
    soft_y_hat, hard_y_hat = output[0], output[1]

    hard_loss = hard_loss_func(y_batch, hard_y_hat)
    soft_loss = soft_loss_func(teacher_model(x_batch),
                               soft_y_hat) if teacher_model and soft_loss_func else None

    loss = alpha * (
        1 if not T_square_make_up else T * T) * soft_loss + beta * hard_loss if teacher_model and soft_loss_func else hard_loss

    # opt非None即在训练集上
    if optimizer is not None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return hard_loss.item(), len(x_batch)


def fit_model(epochs, model, hard_loss_func, opt, train_dl, valid_dl, T,
              teacher_model=None, soft_loss_func=None,
              alpha=0.9, beta=0.1, T_square_make_up=False):
    device = get_available_device()
    for epoch in range(epochs):
        model.train()
        count = 0
        for xb, yb in train_dl:
            count += 1
            loss_batch(model=model.to(device), hard_loss_func=hard_loss_func, x_batch=xb.to(device),
                       y_batch=yb.to(device),
                       optimizer=opt, T=T,
                       teacher_model=teacher_model,
                       soft_loss_func=soft_loss_func, alpha=alpha, beta=beta, T_square_make_up=T_square_make_up)
            if not count % 100:
                print(f"本epoch中第{count}个batch训练完成")

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model.to(device), hard_loss_func, xb.to(device), yb.to(device),
                             teacher_model=teacher_model, soft_loss_func=soft_loss_func,
                             T=T)
                  for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(f"第{epoch}个epoch完成，验证集上损失函数为{val_loss}")

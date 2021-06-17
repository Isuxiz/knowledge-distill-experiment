import torch
import numpy as np


# split是控制是否拆分，因为有soft和hard的两个softmax结果，位置0是soft，1是hard
def loss_batch(model, hard_loss_func, xb, yb, T, teacher_model=None, soft_loss_func=None, opt=None,
               alpha=0.9, beta=0.1, T_square_make_up=False):
    output = model(xb)
    hard_y_hat = output[1]
    hard_loss = hard_loss_func(hard_y_hat, yb)

    soft_loss = soft_loss_func(output[0].log(),
                               teacher_model(xb)) if teacher_model and soft_loss_func else None

    loss = alpha * soft_loss * (1 if not T_square_make_up else T * T) + beta * hard_loss

    # opt非None即在训练集上
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return hard_loss.item(), len(xb)


def fit(epochs, model, hard_loss_func, opt, train_dl, valid_dl, T, teacher_model=None, soft_loss_func=None, alpha=0.9,
        beta=0.1, T_square_make_up=False):
    for epoch in range(epochs):
        model.train()
        count = 0
        for xb, yb in train_dl:
            count += 1
            loss_batch(model=model, hard_loss_func=hard_loss_func, xb=xb, yb=yb, opt=opt, T=T,
                       teacher_model=teacher_model,
                       soft_loss_func=soft_loss_func, alpha=alpha, beta=beta, T_square_make_up=T_square_make_up)
            if not count % 100:
                print(f"本epoch中第{count}个batch训练完成")

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, hard_loss_func, xb, yb, teacher_model=teacher_model, soft_loss_func=soft_loss_func,
                             T=T)
                  for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(f"第{epoch}个epoch完成，验证集上损失函数为{val_loss}")

import os
import torch
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.get_accuracy import get_accuracy

all_infer_result = None

if os.path.exists('./model/all_infer_result.pkl'):
    with open('./model/all_infer_result.pkl', 'rb') as f:
        all_infer_result = pickle.load(f)
        print("读取完成")
else:
    all_infer_result = dict()
    for T in [2, 4, 6, 10, 20]:
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            beta = round(1 - alpha, 1)
            for makeUp in [True, False]:
                config = f'T-{T}_alpha-{alpha}_beta-{beta}_MakeUp-{"T" if makeUp else "F"}'
                config_tuple = (T, alpha, makeUp)
                savePath = f'./model/distilled_models/{config}_distilled'
                # 存在就取训练好的
                if os.path.exists(savePath):
                    print(f'{config}配置下效果：')
                    if torch.cuda.is_available():
                        model = torch.load(savePath)
                    else:
                        model = torch.load(savePath, map_location=torch.device('cpu'))
                    all_infer_result[config_tuple] = get_accuracy(model)
                    print(all_infer_result[config_tuple])
                else:
                    assert True, f'{config} model missed.'
    with open('./model/all_infer_result.pkl', 'wb') as f:
        pickle.dump(all_infer_result, f, pickle.HIGHEST_PROTOCOL)
        print("保存完成")

plt.figure(figsize=(24, 8), dpi=100)
plt.grid(True, linestyle='--', alpha=0.5)
plt.subplot(1, 2, 1)
plt.xlabel("T", fontdict={'size': 16})
plt.ylabel("Accuracy", fontdict={'size': 16})
plt.title("Distill effect in different config (Make-up True)", fontdict={'size': 20})
for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
    beta = round(1 - alpha, 1)
    x = [2, 4, 6, 10, 20]
    y = [all_infer_result[(T, alpha, True)].total_acc for T in x]
    plt.plot(x, y,
             label=f'alpha-{alpha} beta-{beta}')
    plt.scatter(x, y)

plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.xlabel("T", fontdict={'size': 16})
plt.ylabel("Accuracy", fontdict={'size': 16})
plt.title("Distill effect in different config (Make-up False)", fontdict={'size': 20})
for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
    beta = round(1 - alpha, 1)
    x = [2, 4, 6, 10, 20]
    y = [all_infer_result[(T, alpha, False)].total_acc for T in x]
    plt.plot(x, y,
             label=f'alpha-{alpha} beta-{beta}')
    plt.scatter(x, y)

plt.legend(loc='best')

plt.show()

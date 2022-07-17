from torchvision.models import ResNet
from torch import max, sum
from tqdm import tqdm
from torch.utils.data import DataLoader
from time import time
from copy import deepcopy
from torch.optim import Optimizer
from torch.nn.modules import Module


def trainResNetModel(model: ResNet, dataloaders: dict[str, DataLoader], datasize: dict[str, int], criterion: Module, optimizer: Optimizer, num_epochs=25):
    """
    ResNetModelを学習
    """

    best_acc = 0.0

    since = time()

    loss_dict = {"train": [],  "val": []}
    acc_dict = {"train": [],  "val": []}

    for epoch in tqdm(range(num_epochs)):
        if (epoch+1) % 5 == 0:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                # 学習モード
                model.train()
            else:
                # 推論モード
                model.val()

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data

                outputs = model(inputs)
                _, preds = max(outputs.data, 1)

                # 損失
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss * inputs.size(0)
                running_corrects += sum(preds == labels)

            # 精度と損失の平均値
            epoch_loss = running_loss / datasize[phase]
            epoch_acc = running_corrects / datasize[phase]

            # 精度と損失の平均値をhistoryに追加する
            loss_dict[phase].append(epoch_loss)
            acc_dict[phase].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc
            ))

            # 精度が改善した時のみ、モデルの重みを更新する
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())

    time_elapsed = time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))
    print('Best val acc: {:.4f}'.format(best_acc))

    # 重みを反映する
    model.load_state_dict(best_model_wts)

    return model, loss_dict, acc_dict

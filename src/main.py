from torchvision.datasets import Food101
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, ToTensor, Normalize
from torch.utils.data import random_split, DataLoader
from constants.path import APP_DIR, DATA_DIR
from torchvision.models import resnet18
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch

from model.resnet import trainResNetModel

# 訓練データ 前処理
transForTrain = Compose([
    # 256×256にリサイズ
    Resize((256, 256)),

    # 左右逆転
    RandomHorizontalFlip(),

    # テンソル化
    ToTensor(),

    # 平均値と標準偏差で正規化
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# テストデータ 前処理
transForTest = Compose([
    # 256×256にリサイズ
    Resize((256, 256)),

    # テンソル化
    ToTensor(),

    # 平均値と標準偏差で正規化
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# 訓練データ
trainFood101 = Food101(root=DATA_DIR, split='train', transform=transForTrain)

# テストデータ
testFood101 = Food101(root=DATA_DIR, split='test', transform=transForTest)

# 分割サイズ
splitRatio = 0.8

# 訓練データサイズ
trainSize = int(splitRatio * len(trainFood101))

# 検証データサイズ
valSize = len(trainFood101) - trainSize

print(trainSize, valSize)

# データ
trainData, valData = random_split(trainFood101, [trainSize, valSize])

# バッチサイズ
batchSize = 1

# データローダー
trainLoader, valLoader = [
    DataLoader(trainData, batch_size=batchSize, shuffle=True),
    DataLoader(valData, batch_size=batchSize, shuffle=False)
]

# ResNet18
model = resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = Linear(512, 101)

# 学習の設定
lr = 1e-4
epoch = 10
optim = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
criterion = CrossEntropyLoss()

# 訓練
fittedModel, loss, acc = trainResNetModel(
    model,
    dataloaders={"train": trainLoader, "val": valLoader},
    datasize={"train": trainSize, "val": valSize},
    criterion=criterion,
    optimizer=optim,
    num_epochs=epoch
)

print(fittedModel)
print(acc)
print(loss)

torch.save(fittedModel.state_dict(), APP_DIR + 'build')

# 可視化
# loss_train = loss["train"]
# loss_val = loss["val"]

# acc_train = acc["train"]
# acc_val = acc["val"]

# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# axes[0].plot(range(epoch), loss_train, label="train")
# axes[0].plot(range(epoch), loss_val,    label="val")
# axes[0].set_title("Loss")
# axes[0].legend()

# axes[1].plot(range(epoch), acc_train, label="train")
# axes[1].plot(range(epoch), acc_val,    label="val")
# axes[1].set_title("Acc")
# axes[1].legend()

# fig.tight_layout()
# fig.savefig(APP_DIR + "/img.png")

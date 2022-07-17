from torchvision.datasets import Food101
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, ToTensor, Normalize
from torch.utils.data import random_split, DataLoader
from constants.path import DATA_DIR
from torchvision.models import resnet18
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import Adam

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
epoch = 2
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

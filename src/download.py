from torchvision.datasets import Food101
from constants.path import DATA_DIR

# 訓練データ
trainFood101 = Food101(root=DATA_DIR, split='train', download=True)

# テストデータ
testFood101 = Food101(root=DATA_DIR, split='test', download=True)

print('train', trainFood101)
print('test', testFood101)

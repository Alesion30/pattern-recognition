from PIL import Image
import numpy as np

from constants.path import FOOD_101_DIR

img = np.array(Image.open(FOOD_101_DIR + '/images/apple_pie/134.jpg'))
print(img)

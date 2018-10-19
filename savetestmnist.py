import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import TensorBoard
from time import time
import numpy as np
import random

batch_size = 128
num_classes = 10
epochs =  1
save_model = True

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

x_train = np.ceil(x_train)
x_test = np.ceil(x_test)

def addnoise(data,prob):
  print random.random()
  for k in range(data.shape[0]):
    for i in range(data.shape[1]):
      for j in range(data.shape[2]):
        if random.random() <= prob:
          data[k][i][j] = 1 if data[k][i][j] == 0 else 0


addnoise(x_train,0.2)
print x_train[0]
import scipy.misc
scipy.misc.imsave('outfile.jpg', x_train[0].reshape((28,28)))
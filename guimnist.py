from tkinter import *
import numpy as np
import keras
from keras.datasets import mnist
from keras import backend as K
from keras.models import model_from_json
import math

mat = np.zeros((28, 28), dtype=np.float32)
print(mat[0][0])
drawedpixels = []

tkroot = Tk()
canvas_width = 300
canvas_height = 480
widget = Canvas(tkroot, width=canvas_width, height=canvas_height)
widget.pack(expand=YES, fill=BOTH)

size = 10


def callback(event):
    ezerx = event.x / 10 - 1
    ezery = event.y / 10 - 1
    addpixel(ezerx, ezery)
    if event.x > 80 and event.x < 220 and event.y > 295 and event.y < 325:
        reset()


def onLeftDrag(event):
    ezerx = event.x / 10 - 1
    ezery = event.y / 10 - 1
    print(ezerx)
    addpixel(ezerx, ezery)


def addpixel(x, y):
    x = math.ceil(x)
    y = math.ceil(y)
    if x < 28 and y < 28 and x >= 0 and y >= 0:
        if mat[x][y] == 0:
            drawedpixels.append(
                widget.create_rectangle(size * (x + 1), size * (y + 1), size * (x + 2), size * (y + 2), fill='black'))
            mat[x][y] = 1
            classify()


def drawcanvas():
    for i in range(30):
        widget.create_line(size, i * size, 280 + size, i * size, width=1)
        widget.create_line(i * size, size, i * size, 280 + size, width=1)

    global text
    widget.create_rectangle(100, 300, 200, 320, fill='#3f3f3f')
    widget.create_text((150, 310), text="reset", fill="white")
    text = widget.create_text(100, 340, text="start drawing a digit")



def reset():
    global mat
    mat = np.zeros((28, 28), dtype=np.float32)
    for pixel in drawedpixels:
        widget.delete(pixel)


batch_size = 128
num_classes = 10
epochs = 1

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

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                     metrics=['accuracy'])


def classify():
    pre = mat
    pre = np.flip(pre, 1)
    pre = np.rot90(pre)
    pre = pre.reshape([1, 28, 28, 1])
    prediction = loaded_model.predict(pre)
    pre2 = prediction.argmax(axis=-1)[0]
    textezer = str(pre2) + " - " + str(int(round(prediction[0][pre2], 2) * 100)) + "%"
    widget.itemconfig(text, text=textezer, font=("Purisa", 20))


drawcanvas()
widget.bind('<B1-Motion>', onLeftDrag)
widget.bind("<Button-1>", callback)
widget.focus()
tkroot.title('Handwritten digits classifier')
tkroot.mainloop()

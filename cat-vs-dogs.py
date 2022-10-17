# Importing packages

import pandas as pd
import numpy as np
import os
from zipfile import ZipFile as zipper
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from tqdm import tqdm

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

os.listdir('./dogs-vs-cats')

train_path = './dogs-vs-cats/train.zip'
test_path = './dogs-vs-cats/test1.zip'

destination = '/kaggle/files/images'

with zipper(train_path, 'r') as zipp:
    zipp.extractall(destination)
    
with zipper(test_path, 'r') as zipp:
    zipp.extractall(destination)


#defining pandas dataframe adding a binary and a categorical label for each image.

train = pd.DataFrame({'file': os.listdir('/kaggle/files/images/train')})
labels = []
binary_labels = []
for i in os.listdir('/kaggle/files/images/train'):
    if 'dog' in i:
        labels.append('dog')
        binary_labels.append(1)
    else:
        labels.append('cat')
        binary_labels.append(0)

train['labels'] = labels
train['binary_labels'] = binary_labels
test = pd.DataFrame({'file': os.listdir('/kaggle/files/images/test1')})

train.head()
test.head()

#only plot some sample images from train dat
filepath = '/kaggle/files/images/train/'
fig = plt.figure(1, figsize = (20, 20))
for i in range(10):
    plt.subplot(5, 5, i + 1)
    pic = PIL.Image.open(filepath + os.listdir(filepath)[i])
    plt.imshow(pic)
    plt.axis('off')

plt.show()


#split the training and test data sets by 90% and 10%
train_set, val_set = train_test_split(train,
                                     test_size=0.1)
print(len(train_set), len(val_set))

#apply pre-processing to each images -brighten up-
for i in tqdm(range(len(os.listdir(filepath)))):
        pic_path = filepath + os.listdir(filepath)[i]
        pic = PIL.Image.open(pic_path)
        pic_sharp = pic.filter(PIL.ImageFilter.UnsharpMask(radius=2, percent=100))
        pic_sharp.save(pic_path)
        

#ImageDataGenerator converts one image to a batch of randomly transformed image
#from that same image -zoomed in, rotated, cropped- 
#to get more data in the dataset from one image ! 
#Also called Image Augmentation

train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)
batch_size = 128


train_generator = train_gen.flow_from_dataframe(
    dataframe = train_set,
    directory = destination + '/train/',
    x_col = 'file',
    y_col = 'labels',
    class_mode = 'categorical',
    target_size = (224,224),
    batch_size = batch_size
)


validation_generator = val_gen.flow_from_dataframe(
    dataframe = val_set,
    directory = destination + '/train/',
    x_col = 'file',
    y_col = 'labels',
    class_mode = 'categorical',
    target_size = (224,224),
    batch_size = batch_size,
    shuffle = False
)


data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(224,
                                  224,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)


#build model, changed optimizer, learning rate, number of layers, 
#activation function -Relu, SoftMax, LeakyRelu, PReLu, ELU, ThresholdedReLU-.
input_shape = (224, 224, 3)
n_class = 2
korobka = Sequential([
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu'),

    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='elu'),


    layers.Flatten(),

    layers.Dropout(0.5),
    layers.Dense(2, activation = 'softmax')
])

#"Adam is faster to converge. SGD is slower but generalizes better." 
#Adam default learning rate is 0.001, SGD is 0.01
learning_rate = "0.002"

#learning rate changed here
korobka.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


early_stopping = EarlyStopping(
    monitor = "val_loss",
    patience = 3,
    verbose = 1,
    mode = "min")

#train model : changed epoch here
with tf.device('/device:GPU:0'):
    epochs=30
    history = korobka.fit_generator(train_generator, 
                            validation_data=validation_generator, 
                            epochs=epochs,
                            validation_steps = val_set.shape[0] // batch_size,
                            steps_per_epoch = train_set.shape[0] // batch_size, callbacks = [early_stopping])
    
#ici produce cost et accucary graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(28)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

with tf.device('/device:GPU:0'):
    val_pred = korobka.predict(validation_generator, steps = np.ceil(val_set.shape[0] / batch_size))
    
val_set['normalpreds'] = np.argmax(val_pred, axis = -1)
labels = dict((v,k) for k,v in train_generator.class_indices.items())

val_set['normalpreds'] = val_set['normalpreds'].map(labels)

#ici confusion matrix
fig, ax = plt.subplots(figsize = (9, 6))

cm = confusion_matrix(val_set["labels"], val_set["normalpreds"])

disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["cat", "dog"])
disp.plot(cmap = plt.cm.Blues, ax = ax)

ax.set_title("Validation Set")
plt.show()

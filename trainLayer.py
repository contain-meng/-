import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from keras.models import load_model
import numpy as np
from keras.utils import plot_model
import matplotlib.pyplot as plt
os.environ["PATH"] += os.pathsep +'C:\Program Files (x86)\Graphviz2.38\bin'

train_dir = r'.\newDataSet\train'
val_dir = r'.\newDataSet\val'
test_dir = r'.\newDataSet\test'

train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=128,
    class_mode='categorical'
)
validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=128,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=128,
    class_mode='categorical'
)

model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
                    activation='relu',
                    padding='same', input_shape=(48, 48, 3)))   #输入3通道48*48的像素图像,卷积核为5*5
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(256, kernel_size=(5, 5), strides=(1, 1),
                     padding='same',
                     activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dense(7, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,   #多分类问题
                  optimizer=keras.optimizers.rmsprop(lr=0.0001),  # need to train a lot of epochs
                  metrics=['accuracy'])

history = model.fit_generator(
        train_generator,
        steps_per_epoch=650,
        epochs=40,
        validation_data=validation_generator,
        validation_steps=32
    )
# model.save_weights('CNN_Model_weights.h5')
model.save('new_model.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.figure("acc")
plt.plot(epochs, acc, 'r-', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='validation acc')
plt.title('The comparision of train_acc and val_acc')
plt.legend()
plt.show()

plt.figure("loss")
plt.plot(epochs, loss, 'r-', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('The comparision of train_loss and val_loss')
plt.legend()
plt.show()

plot_model(model, to_file="cnn_model.png", show_shapes=True, show_layer_names="True")

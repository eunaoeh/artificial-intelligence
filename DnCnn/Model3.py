import tensorflow as tf
from tensorflow.keras.layers import GaussianNoise, Conv2D, Activation, Add, BatchNormalization
from tensorflow.keras import Input
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image

(x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

x_train, x_test = x_train/255.0, x_test/255.0
y_train, y_test = x_train, x_test

# Model 3
inputs = Input(shape=(None,None,3))
data = GaussianNoise(0.1) (inputs)
x = Conv2D(64, (3, 3), padding='same') (data)
x = BatchNormalization() (x)
x = Activation('relu') (x)
x = Conv2D(64, (3, 3), padding='same') (x)
x = BatchNormalization() (x)
x = Activation('relu') (x)
x = Conv2D(64, (3, 3), padding='same') (x)
x = BatchNormalization() (x)
x = Activation('relu') (x)
x = Conv2D(64, (3, 3), padding='same') (x)
x = BatchNormalization() (x)
x = Activation('relu') (x)
x = Conv2D(3, (3, 3), padding='same') (x)
outputs = Add() ([x, data])
model = Model(inputs=inputs, outputs=outputs)

model.summary()
optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='MSE', optimizer=optimizer, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=32)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

#Inference Noisy.png
img = np.asarray(Image.open('noisy.png'))
img = np.expand_dims(img, axis=0)
img = img.astype('float32')/255.0
data = model.predict(img)
data = np.squeeze(data, axis=0)
result = tf.keras.preprocessing.image.array_to_img(data)
result.save('./Model3.png')

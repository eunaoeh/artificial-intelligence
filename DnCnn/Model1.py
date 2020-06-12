import tensorflow as tf
from tensorflow.keras.layers import GaussianNoise, Conv2D, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input
import numpy as np
from PIL import Image

(x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

x_train, x_test = x_train.astype('float32')/255.0, x_test.astype('float32')/255.0
y_train, y_test = x_train, x_test

# Model 1
inputs = Input(shape=(None,None,3))
x = GaussianNoise(0.1) (inputs)
x = Conv2D(64, (3, 3), padding='same') (x)
x = Activation('relu') (x)
x = Conv2D(64, (3, 3), padding='same') (x)
x = Activation('relu') (x)
x = Conv2D(64, (3, 3), padding='same') (x)
x = Activation('relu') (x)
x = Conv2D(64, (3, 3), padding='same') (x)
x = Activation('relu') (x)
x = Conv2D(3, (3, 3), padding='same') (x)
model = Model(inputs=inputs, outputs=x)

optimizer= tf.keras.optimizers.Adam()
model.compile(loss='MSE', optimizer=optimizer, metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=100, batch_size=32)

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

#Inference Noisy.png
img = np.asarray(Image.open('noisy.png'))
img = np.expand_dims(img, axis=0)
img = img.astype('float32')/255.0
data = model.predict(img)
data = np.squeeze(data, axis=0)
result = tf.keras.preprocessing.image.array_to_img(data)
result.save('./Model1.png')

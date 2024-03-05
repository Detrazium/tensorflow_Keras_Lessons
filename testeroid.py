import os, logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import time

import keras
from keras import Sequential
from keras.datasets import mnist
import keras.backend as K
import tensorflow as tf
from keras import initializers
from keras.layers import (
Dense,
Flatten,
BatchNormalization,
Dropout,
concatenate,
Reshape,
Conv2D,
Conv2DTranspose,
LeakyReLU)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[y_train == 7]
y_train = y_train[y_train == 7]

BUFFER_SIZE = x_train.shape[0]
BATCH_SIZE = 100

BUFFER_SIZE = BUFFER_SIZE // BATCH_SIZE * BATCH_SIZE
x_train = x_train[:BUFFER_SIZE]
y_train = y_train[:BUFFER_SIZE]
print(x_train.shape, y_train.shape)

x_train = x_train / 255
x_test = x_test / 255

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

train_dadasss = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

hidd_dim = 2
def drop_and_batch():
	return Dropout(0.3)(BatchNormalization())

def Generator():
	generator = Sequential([
		Dense(7 * 7 * 256, activation='relu', input_shape=(hidd_dim,)),
		BatchNormalization(),
		Reshape((7, 7, 256)),
		Conv2DTranspose(128, (5, 5), strides= (1, 1), padding='same', activation='relu'),
		BatchNormalization(),
		Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu'),
		BatchNormalization(),
		Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='sigmoid')
	])
	return generator

def Diskret():
	discriminator = Sequential([
		Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape= [28, 28, 1]),
		LeakyReLU(),
		Dropout(0.3),
		Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
		LeakyReLU(),
		Dropout(0.3),
		Flatten(),
		Dense(1)
	])
	return discriminator
generator = Generator()
discriminator = Diskret()

cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
def generator_loss(fake_outer):
	loss = cross_entropy(tf.ones_like(fake_outer), fake_outer)
	return loss
def diskret_loss(real_outer, fake_outer):
	real_loss = cross_entropy(tf.ones_like(real_outer), real_outer)
	fake_loss = cross_entropy(tf.zeros_like(fake_outer), fake_outer)
	total_loss = real_loss + fake_loss
	return total_loss

generator_optimizer = keras.optimizers.Adam(1e-4)
discriminator_optima = keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images):
	noise = tf.random.normal([BATCH_SIZE, hidd_dim])
	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		gen_images = generator(noise, training=True)

		real_outer = discriminator(images, training=True)
		fake_outer = discriminator(gen_images, training=True)

		gen_losss = generator_loss(fake_outer)
		disc_loss = diskret_loss(real_outer, fake_outer)

	gradient_of_gen = gen_tape.gradient(gen_losss, generator.trainable_variables)
	gradient_of_discrim = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(gradient_of_gen, generator.trainable_variables))
	discriminator_optima.apply_gradients(zip(gradient_of_discrim, discriminator.trainable_variables))

	return gen_losss, disc_loss

def TRAIN(dataset, epochs):
	history = []
	MAX_PRINT_LAB = 10
	ugh = BUFFER_SIZE // (BATCH_SIZE * MAX_PRINT_LAB)

	for epoch in range(1, epochs + 1):
		print(f'{epoch} / {EPOCHS}: ', end='')

		start = time.time()
		n = 0
		gen_loss_epoch = 0
		for image_batch in dataset:
			gen_losss, disc_loss = train_step(image_batch)
			gen_loss_epoch += np.mean(gen_losss)
			if (n % ugh == 0): print('=', end='')
			n+=1

		history += [gen_loss_epoch / n]
		print("; "+ str(history[-1]))
		print('Время исполнения эпохи {} пришлось на {} секунд'.format(epoch, time.time() - start))
	return history
EPOCHS = 20

history = TRAIN(train_dadasss, EPOCHS)

plt.plot(history)
plt.grid(True)
plt.show()

n = 2
total = 2 * n + 1
plt.figure(figsize=(total, total))

num = 1
for i in range(-n, n + 1):
	for j in range(-n, n + 1):
		xxx = plt.subplot(total, total, num)
		num += 1
		img = generator.predict(np.expand_dims([0.5 * i / n, 0.5 * j / n], axis=0))
		plt.imshow(img[0, :, :, 0], cmap='gray')
		xxx.get_xaxis().set_visible(False)
		xxx.get_yaxis().set_visible(False)
plt.show()


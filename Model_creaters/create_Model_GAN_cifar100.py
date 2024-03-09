import os, logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
from cifar100_classification_diction import Class_cifar
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import numpy as np
import keras
from keras import Sequential
from keras.datasets import cifar100, mnist
from keras.layers import (
Dense,
Conv2D,
Conv2DTranspose,
Flatten,
Dropout,
BatchNormalization,
Reshape,
LeakyReLU
)

class Mygan_N_1():
	"""/////////////////////////////

		Класс формирующий генеративно-состязательную сеть на cifar100

		1. Решить проблему резкого переобучения
		2. заново обучить

		///////////////////////////"""
	def __init__(self):

		self.generator = self.Generator()
		self.discriminator = self.Discriminator()
		self.Cross_Entropy = keras.losses.BinaryCrossentropy(from_logits=True)
		self.gen_optimizator = keras.optimizers.Adam(1e-4)
		self.disc_optima = keras.optimizers.Adam(1e-4)
	def D_AND_B(self, x):
		return Dropout(0.3)(BatchNormalization()(x))

	def Generator(self):
		generator = Sequential([
			Dense(8 * 8 * 256, activation='relu', input_shape=(2,)),
			BatchNormalization(),
			# Dropout(0.3),
			Reshape((8, 8, 256)),
			Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', activation='relu'),
			# Dropout(0.5),
			BatchNormalization(),
			Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu'),
			# Dropout(0.3),
			BatchNormalization(),
			Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='sigmoid')
		], name='Generator')
		generator.summary()
		return generator
	def Discriminator(self):
		discriminator = Sequential([
			Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[32, 32, 3]),
			LeakyReLU(),
			Dropout(0.3),
			Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
			LeakyReLU(),
			Dropout(0.3),
			Flatten(),
			Dense(1)
		], name='Discriminator')

		discriminator.summary()
		return discriminator
	def Dataset(self, key = None):
		(x_train, y_train), (x_test, y_test) = cifar100.load_data()
		if key ==  None:
			x = []
			classes = Class_cifar(key).Get()
			print(classes)
			for i, el2 in enumerate(y_train):
				if el2 == key:
					x.append(x_train[i])
			x_train = np.array(x)
			BATCH_SIZE = 10
		else:
			BATCH_SIZE = 100
		BUFFER_SIZE = x_train.shape[0]
		BUFFER_SIZE = BUFFER_SIZE // BATCH_SIZE * BATCH_SIZE
		x_train = x_train[:BUFFER_SIZE]
		x_train = x_train / 255
		x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
		dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
		return dataset, BUFFER_SIZE, BATCH_SIZE
	def gen_loss(self, ff):
		loss = self.Cross_Entropy(tf.ones_like(ff), ff)
		return loss
	def dis_loss(self, rr, ff):
		real = self.Cross_Entropy(tf.ones_like(rr), rr)
		fake = self.Cross_Entropy(tf.zeros_like(ff), ff)
		total = real + fake
		return total

	@tf.function
	def Method_one_train(self, images):
		Notna = tf.random.normal([100, 2])

		with tf.GradientTape() as generatort_type, tf.GradientTape() as discriminant_type:
			gen_imm = self.generator(Notna, training=True)

			real_outer = self.discriminator(images, training=True)
			fake_outer = self.discriminator(gen_imm, training=True)

			gen_losses = self.gen_loss(fake_outer)
			dis_losses = self.dis_loss(real_outer, fake_outer)

		g_of_gen = generatort_type.gradient(gen_losses, self.generator.trainable_variables)
		g_of_dis = discriminant_type.gradient(dis_losses, self.discriminator.trainable_variables)

		self.gen_optimizator.apply_gradients(zip(g_of_gen, self.generator.trainable_variables))
		self.disc_optima.apply_gradients(zip(g_of_dis, self.discriminator.trainable_variables))
		return gen_losses, dis_losses

	def history_plot(self, history):
		plt.plot(history)
		plt.grid(True)
		plt.show()

		n = 2
		total= 2 * n +1
		plt.figure(figsize=(total, total))
		num = 1
		for i in range(-n, n + 1):
			for jj in range(-n, n + 1):
				xpl = plt.subplot(total, total, num)
				num += 1
				imm = self.generator.predict(np.expand_dims([0.5 * i / n, 0.5* jj / n], axis= 0))
				plt.imshow(imm[0, :, :, 0], cmap='gray')
				xpl.get_xaxis().set_visible(False)
				xpl.get_yaxis().set_visible(False)
		plt.show()
	def Train(self, EPOCHS = 10, Num_class = None):
		dataset, BUFFER_SIZE, BATCH_SIZE = self.Dataset(key = Num_class)
		history = []
		M_PRINT_LB =10
		ugh = BUFFER_SIZE // (BATCH_SIZE * M_PRINT_LB)
		print(ugh)

		for ep in range(1, EPOCHS + 1):
			print(f'{ep} / {EPOCHS}: ', end='')

			start = time.time()
			n = 0
			gen_L_EP = 0
			for image_BB in dataset:
				gen_losses, disc_losses = self.Method_one_train(image_BB)
				gen_L_EP += np.mean(gen_losses)
				if (n % ugh == 0): print('=', end='')
				n+=1
			history += [gen_L_EP/n]
			print('>: '+str(history[-1]))
			print('Epoch time {} in {} second'.format(ep, time.time()- start))
		return self.history_plot(history)
	def model_saving(self):
		"""Saving Model"""
		self.generator.save("Generator''Mygan_N_1()''.h5")
		self.discriminator.save('Discriminator""Mygan_N_1()"".h5')
def main():
	gen = Mygan_N_1()
	gen.Train(EPOCHS = 10, Num_class = 19)
	# gen.model_saving()

if __name__ == '__main__':
	main()





import os, logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
import keras
import numpy as np
from keras.datasets import cifar100
import matplotlib.pyplot as plt
from keras.activations import exponential
from keras.layers import (
Input,
Dense,
Dropout,
BatchNormalization,
Lambda,
concatenate,
Flatten,
Reshape
)
hd = 2
batch_size = 50

class Autoencoder():
	"""Класс формирующий автоэнкодер.
	{ПОКА НЕ РАБОТАЕТ}
	Непонятный мне баг на слое лямбда
	возвращающий 'H' при первой эпохе обучения
	в виде не нужного тензора формы (50, 2)-То что ожидается
	а с лишней единицей ({1}, 50, 2)-То что по какой то причине получаем"""

	def __init__(self):
		self.model = self.Create_Model()

	def Create_Model(self):
		"""Сама модель Автоэнкодера
		Формирование модели слоев Энкодера, Математическое ожидание,
		дисперсия"""

		def Drop_and_Batcher(x):
			"""Функция благотворного объединения Dropout и BatchNormalization"""
			return Dropout(0.3)(BatchNormalization()(x))

		inputer_image = Input(batch_shape=(batch_size, 32, 32, 3), name='Inputerion')
		Flt = Flatten()(inputer_image)
		LayEr = Dense(288, activation='relu')(Flt)
		LayEr = Drop_and_Batcher(LayEr)
		LayEr = Dense(144, activation='relu')(LayEr)
		LayEr = Drop_and_Batcher(LayEr)
		L_mathW = Dense(hd)(LayEr),
		L_LoggerD = Dense(hd)(LayEr)

		def Center_quad(args):
			"""Вспомогательная функция среднего звена 'бутылочного горлышка'."""
			global L_mathW, L_LoggerD
			L_mathW, L_LoggerD = args
			Normal = (keras.initializers.random_normal(mean=0., stddev=1.0)(shape=(batch_size, hd)))
			print((exponential(L_LoggerD / 2) * Normal), "|UNIT|", L_mathW, "|UNIT2|")
			CNTH = exponential(L_LoggerD / 2) * Normal + L_mathW
			return CNTH

		def Center_layer():
			"""Возвращает средний слой между енкодером и декодером"""
			H = Lambda(Center_quad, output_shape=(hd,))([L_mathW, L_LoggerD])
			return H

		def Decoder():
			"""Формирование Декодера и выходного значения"""
			inputer_decoder = Input(shape=(hd,))
			dd = Dense(144, activation='relu')(inputer_decoder)
			dd = Drop_and_Batcher(dd)
			dd = Dense(288, activation='relu')(dd)
			dd = Drop_and_Batcher(dd)
			dd = Dense(32 * 32 * 3, activation='sigmoid')(dd)
			decoded = Reshape((32, 32, 3))(dd)
			return inputer_decoder, decoded

		def compile_model(inputer_image):
			"""Склейка модели во что то внятное"""

			inputer_decoder, decoded = Decoder()
			H = Center_layer()

			encoder = keras.Model(inputer_image, H, name='encoderOne')
			decoder = keras.Model(inputer_decoder, decoded, name='decodedOne')

			model = keras.Model(inputer_image, decoder(encoder(inputer_image)), name='autoencoder')
			model.summary()
			return model
		return compile_model(inputer_image)

	def Losses(self, x, y):
		"""Функция потерь по дивергенции Кульбака-Лейблера"""
		x = tf.reshape(x, shape=(batch_size, 32, 32))
		y = tf.reshape(x, shape=(batch_size, 32, 32))
		loss = tf.reduce_sum(tf.square(x - y))
		KL_loss = -0.5 * tf.reduce_sum(1 + L_LoggerD - tf.square(L_mathW) - exponential(L_LoggerD), axis=-1)
		return loss + KL_loss

	def Train_MM(self, loadd):
		"""Обучение"""
		x_train = loadd
		self.model.compile(optimizer='adam', loss=self.Losses)
		self.model.fit(x_train, x_train, batch_size=50, epochs=10, shuffle=True)
		return self.model

def loaded():
	"""Загружаю и стандартизирую обучающую выборку по базе изображений: Cifar-100"""
	(x_train, y_train), (x_test, y_test) = cifar100.load_data()

	x_train = x_train.astype('float32') / 255
	# x_test = x_test / 255

	x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
	# x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))
	return x_train


def Testing():
	"""Тесты"""
	model = Autoencoder().Train_MM(loaded())
def main():
	Testing()

if __name__ == '__main__':
	main()





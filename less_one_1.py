import os, logging
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import keras
from keras import initializers
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import to_categorical, pad_sequences
from keras.activations import exponential
from keras.layers import (
Dense,
GRU,
Bidirectional,
SimpleRNN,
Dropout,
TextVectorization,
Embedding,
Conv2D,
UpSampling2D,
InputLayer,
SimpleRNN,
Input,
Flatten,
MaxPooling2D,
LSTM,
Reshape,
Lambda,
BatchNormalization,
concatenate)
import re
import textwrap
import numpy as np
import tensorflow as tf

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


"""Мои уроки но НС на керасе"""
class Neuron_keras_lessons:
	def __init__(self):
		key = input("Press lesson number: lessons -- (1, 2, 3, 4, 5, 6 ...)|| works -- (print: my) \nPress:  ")
		if key == "1":
			self.Neuron_Keras_test_one()
		elif key == "2":
			self.Neuron_Keras_test_two()
		elif key == '3':
			self.swert_Keras_test_three()
		elif key == '4':
			self.less_work_four()
		elif key == '5':
			self.Less_work_five()
		elif key == '6':
			self.less_work_six()
		elif key == '7':
			self.Less_work_seven()
		elif key == '8':
			self.less_work_eich()
		elif key == '9':
			self.less_work_nine_BIRECTIONAL()
		elif key == '10':
			self.less_work_ten_Autoencoder()
		elif key == '11':
			self.less_work_oneten_RASH_Autoencoder()

		elif key == "my":
			self.My_Neuron_test()
		else:
			print("lesson nothing")

	def Neuron_Keras_test_one(self):
		i = np.array([-40, -10, 0, 8, 15, 22, 38])
		w = np.array([-40, 14, 32, 46, 59, 72, 100])

		Nmode = keras.Sequential()
		Nmode.add(Dense(units=1, input_shape=(1,), activation='linear'))
		Nmode.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))

		log = Nmode.fit(i, w, epochs=500, verbose=False)

		print(Nmode.get_weights())

		plt.plot(log.history['loss'])
		plt.grid(True)
		plt.show()

		print("ENDED TRAIN")
	def Neuron_Keras_test_two(self):
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

		x_train = x_train / 255
		x_test = x_test / 255

		y_train_cat = keras.utils.to_categorical(y_train, 10)
		y_test_cat = keras.utils.to_categorical(y_test, 10)

		size_v = 5000
		x_val_spl = x_train[:size_v]
		y_val_spl = y_train_cat[:size_v]

		x_train_spl = x_train[size_v:]
		y_train_spl = y_train_cat[size_v:]

		def first_25_nubers():
			plt.figure(figsize=(10, 5))
			for i in range(25):
				plt.subplot(5,5,i+1)
				plt.xticks([])
				plt.yticks([])
				plt.imshow(x_train[i], cmap=plt.cm.binary)
			plt.show()

		Model_Neuron = keras.Sequential([
			Flatten(input_shape=(28, 28, 1)),
			Dense(150, activation='relu'),
			Dense(200, activation='relu'),
			Dense(350, activation='relu'),
			Dense(50, activation='relu'),
			Dense(10, activation='softmax')
		])

		my_ad = keras.optimizers.Adam(learning_rate=0.0001)
		My_optima = keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=True)



		Model_Neuron.compile(optimizer=My_optima,
							 loss='categorical_crossentropy',
							 metrics=['accuracy'])

		# Model_Neuron.fit(x_train, y_train_cat, batch_size=30, epochs=5, validation_split=0.2)

		x_train_spl, x_val_spl, y_train_spl, y_val_spl = train_test_split(x_train, y_train_cat, test_size= 0.2)



		Model_Neuron.fit(x_train_spl, y_train_spl, batch_size=32, epochs=5, validation_data=(x_val_spl, y_val_spl))

		Model_Neuron.evaluate(x_test, y_test_cat)


		for n in range(100, 1000, 50):
			x = np.expand_dims(x_test[n], axis=0)
			res = Model_Neuron.predict(x)
			print(res)
			print(f"Распознанный номер: {np.argmax(res)}")

			plt.imshow(x_test[n], cmap=plt.cm.binary)
			plt.show()


		predd = Model_Neuron.predict(x_test)
		predd = np.argmax(predd,axis=1)

		maskes = predd == y_test
		print(maskes[:10])
		x_false = x_test[~maskes]
		p_false = predd[~maskes]
		print(x_false.shape)

		for i in range(5):
			print("Значение сетевой сети: " +str(p_false[i]))
			plt.imshow(x_false[i], cmap=plt.cm.binary)
			plt.show()

	def swert_Keras_test_three(self):
		def Neuroninsm():
			(x_train, y_train), (x_test, y_test) = mnist.load_data()

			x_train = x_train / 255
			x_test = x_test / 255

			y_train_cat = keras.utils.to_categorical(y_train, 10)
			y_test_cat = keras.utils.to_categorical(y_test, 10)

			x_train = np.expand_dims(x_train, axis=3)
			x_test = np.expand_dims(x_test, axis=3)

			print(x_train.shape)

			Model = keras.Sequential([
				Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
				MaxPooling2D((2, 2), strides=2),
				Conv2D(64, (3, 3), padding='same', activation='relu'),
				MaxPooling2D((2, 2), strides=2),
				Flatten(),
				Dense(128, activation='relu'),
				Dense(10, activation='softmax')
			])
			print(Model.summary())

			Model.compile(optimizer='adam',
						  loss='categorical_crossentropy',
						  metrics=['accuracy'])
			itog = Model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

			Model.evaluate(x_test, y_test_cat)
		Neuroninsm()
	def less_work_four(self):
		with (open(r'C:\Users\Антонио\PycharmProjects\Neuron+sett\pictures\m_Test_one .jpg', 'rb') as fff,
			  open(r'C:\Users\Антонио\PycharmProjects\Neuron+sett\pictures\m_Style.jpg', 'rb') as kkk):
			one = fff.read()
			twow = kkk.read()
			img_picture = Image.open(BytesIO(one))
			img_style = Image.open(BytesIO(twow))

		plt.subplot(1, 2, 1)
		plt.imshow(img_picture)
		plt.subplot(1, 2, 2)
		plt.imshow(img_style)
		plt.show()

		x_imm = keras.applications.vgg19.preprocess_input(np.expand_dims(img_picture, axis=0))
		x_stt = keras.applications.vgg19.preprocess_input(np.expand_dims(img_style, axis=0))

		def deprocess_img(proc_img):
			i = proc_img.copy()
			if len(i.shape) == 4:
				i = np.squeeze(i, 0)
			assert len(i.shape) == 3, ('input to deprocess image must be an image of',
									   'dimension[1, height, widt, channel] or [height, width, channel]')
			if len(i.shape) != 3:
				raise ValueError("invalid input to deprocessing image")

			i[:, :, 0] += 103.939
			i[:, :, 1] += 116.779
			i[:, :, 2] += 123.68
			i = i[:, :, ::-1]

			i = np.clip(i, 0, 255).astype('uint8')
			return i

		content_l = ['block5_conv2']

		style_l = ['block1_conv1',
				   'block2_conv1',
				   'block3_conv1',
				   'block4_conv1',
				   'block5_conv1'
				   ]
		num_c_l = len(content_l)
		num_s_l = len(style_l)

		vgg = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
		vgg.trainable = False

		style_out = [vgg.get_layer(name).output for name in style_l]
		content_out = [vgg.get_layer(name).output for name in content_l]
		model_out = style_out + content_out

		print(vgg.input)
		for m in model_out:
			print(m)

		model = keras.models.Model(vgg.input, model_out)
		print(model.summary())

		def get_feat_respesentations(model):
			style_out = model(x_stt)
			content_out = model(x_imm)

			style_feat = [style_l[0] for style_l in style_out[:num_s_l]]
			content_feat = [content_l[0] for content_l in content_out[num_s_l:]]
			return style_feat, content_feat

		def get_cont_loss(base_cont, target):
			return tf.reduce_mean(tf.square(base_cont - target))

		def gramm_M(inp_tens):
			channels = int(inp_tens.shape[-1])
			a = tf.reshape(inp_tens, [-1, channels])
			n = tf.shape(a)[0]
			gram = tf.matmul(a, a, transpose_a=True)
			return gram / tf.cast(n, tf.float32)

		def get_style_l(base_s, gram_target):
			gram_st = gramm_M(base_s)
			return tf.reduce_mean(tf.square(gram_st - gram_target))

		def compute_loss(model, loss_weight, init_image, gram_st_feat, content_feat):
			style_wei, content_wei = loss_weight

			model_out = model(init_image)

			style_out_feat = model_out[:num_s_l]
			content_out_feat = model_out[num_s_l]

			style_score = 0
			content_score = 0

			# Accumulate style losses
			weight_per_style_layer = 1.0 / float(num_s_l)
			for target_sty, comb_sty in zip(gram_st_feat, style_out_feat):
				style_score += weight_per_style_layer * get_style_l(comb_sty[0], target_sty)

			# Accumualte style losses
			weight_per_content_layer = 1.0 / float(num_c_l)
			for target_content, comb_c in zip(content_feat, content_out_feat):
				content_score += weight_per_content_layer * get_cont_loss(comb_c[0], target_content)

			style_score *= style_wei
			content_score *= content_wei

			loss = style_score + content_score
			return loss, style_score, content_score

		num_iter = 100
		cont_weight = 1e3
		style_wei = 1e-2

		style_feature, content_feat = get_feat_respesentations(model)
		gram_style_feat = [gramm_M(style_feat) for style_feat in style_feature]

		init_image = np.copy(x_imm)
		init_image = tf.Variable(init_image, dtype=tf.float32)

		opt = tf.compat.v1.train.AdamOptimizer(learning_rate=2, beta1=0.99, epsilon=1e-1)
		iter_count = 1
		best_loss, best_img = float('inf'), None
		loss_weig = (style_wei, cont_weight)

		cfg = {
			'model': model,
			'loss_weight': loss_weig,
			'init_image': init_image,
			'gram_st_feat': gram_style_feat,
			'content_feat': content_feat
		}
		norm_means = np.array([103.939, 116.779, 123.68])
		min_vals = -norm_means
		max_vals = 255 - norm_means
		imgs = []

		for i in range(num_iter):
			with tf.GradientTape() as tape:
				all_loss = compute_loss(**cfg)

			total_loss = all_loss[0]
			gras = tape.gradient(total_loss, init_image)

			loss, style_score, content_score = all_loss
			opt.apply_gradients([(gras, init_image)])
			clipped = tf.clip_by_value(init_image, min_vals, max_vals)
			init_image.assign(clipped)

			if loss < best_loss:
				best_loss = loss
				best_img = deprocess_img(init_image.numpy())

				plot_img = deprocess_img(init_image.numpy())
				imgs.append(plot_img)
				print(f'Iteration: {i}')

		plt.imshow(best_img)
		plt.show()
		print(best_loss)

	def less_work_five(self):
		with open(r'C:\Users\Антонио\PycharmProjects\Neuron+sett\reccurent\text.txt', 'r', encoding='utf-8') as file:
			text = file.read()
			text = text.replace('\ufeff', '')
			text = re.sub(r'[^А-я" "]', '', text)
		char_num = 34
		tokens = TextVectorization(num_words=char_num, char_level=True)
		tokens.fit_on_texts([text])
		print(tokens.word_index)

		inper = 6
		data = tokens.texts_to_matrix(text)
		mn = data.shape[0] - inper
		X = np.array([data[i:i + inper, :] for i in range(mn)])
		Y = data[inper:]
		print(data.shape)

		model = Sequential()
		model.add(Input((inper, char_num)))
		model.add(SimpleRNN(128, activation='tanh'))
		model.add(Dense(char_num, activation='softmax'))
		model.summary()
		model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
		hist = model.fit(X, Y, batch_size=32, epochs=100)

		def bild_def(inputen, str_len=50):
			for i in range(str_len):
				x = []
				for j in range(i, i + inper):
					x.append(tokens.texts_to_matrix(inputen[j]))
				x = np.array(x)
				inp = x.reshape(1, inper, char_num)
				pred = model.predict(inp)
				d = tokens.index_word[pred.argmax(axis=1)[0]]
				inputen += d
			return inputen

		pep = bild_def('ясно к')
		print(pep)

	def Less_work_seven(self):
		with open(r'C:\Users\Антонио\PycharmProjects\Neuron+sett\pictures\m_Test_one .jpg', 'rb') as file:
			imagee = file.read()
			img = Image.open(BytesIO(imagee))

		def processeed_image(img):
			image = img.resize((256, 256), Image.BILINEAR)
			image = np.array(image, dtype=float)
			size = image.shape
			lab = rgb2lab(1.0 / 255 * image)

			x, y = lab[:, :, 0], lab[:, :, 1:]

			y /= 128
			x = x.reshape(1, size[0], size[1], 1)
			y = y.reshape(1, size[0], size[1], 2)
			return x, y, size

		X, Y, size = processeed_image(img)

		Model = Sequential()
		Model.add(InputLayer(input_shape=(None, None, 1)))
		Model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
		Model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
		Model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
		Model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
		Model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
		Model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
		Model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
		Model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
		Model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
		Model.add(UpSampling2D((2, 2)))
		Model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
		Model.add(UpSampling2D((2, 2)))
		Model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
		Model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
		Model.add(UpSampling2D((2, 2)))

		Model.compile(optimizer='adam', loss='mse')
		Model.fit(x=X, y=Y, batch_size=1, epochs=50)

		with open(r'C:\Users\Антонио\PycharmProjects\Neuron+sett\pictures_test\Stile_TWO.jpg', 'rb') as file2:
			immage = file2.read()
			img2 = Image.open(BytesIO(immage))
		X, Y, size = processeed_image(img2)

		outer = Model.predict(X)
		outer *= 128
		min_vals, max_vals = -128, 127
		ab = np.clip(outer[0], min_vals, max_vals)

		cur = np.zeros((size[0], size[1], 3))
		cur[:, :, 0] = np.clip(X[0][:, :, 0], 0, 100)
		cur[:, :, 1:] = ab
		plt.subplot(1, 2, 1)
		plt.imshow(img)
		plt.subplot(1, 2, 2)
		plt.imshow(lab2rgb(cur))
		plt.show()

	def less_work_eich(self):
		with (open(r'C:\Users\Антонио\PycharmProjects\Neuron+sett\reccurent\fear!!_citait.txt', 'r',
				   encoding='utf=8') as True_c,
			  open(r'C:\Users\Антонио\PycharmProjects\Neuron+sett\reccurent\neg_citait.txt', 'r',
				   encoding='utf=8') as False_c):
			txt_True = True_c.readlines()
			txt_False = False_c.readlines()
			txt_True[0] = txt_True[0].replace('\ufeff', '')
			txt_False[0] = txt_False[0].replace('\ufeff', '')

		txts = txt_True + txt_False
		txts = [i.replace('\n', '') for i in txts]
		count_Tr = len(txt_True)
		count_Fl = len(txt_False)

		maxW = 1080
		tokens = TextVectorization(max_tokens=maxW, standardize='lower_and_strip_punctuation', split='whitespace')
		tokens.adapt(txts)

		maXT = 49
		data = tokens(txts)
		data_dater = pad_sequences(data, maxlen=maXT)

		def Indexer():
			X = data_dater
			Y = np.array([[1, 0]] * count_Tr + [[0, 1]] * count_Fl)
			indd = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
			X = X[indd]
			Y = Y[indd]
			return X, Y

		def Model_create():
			Model = Sequential([
				Embedding(maxW, 150, input_shape=(10,)),
				LSTM(128, activation='tanh', return_sequences=True),
				LSTM(64, activation='tanh'),
				Dense(2, activation='softmax')
			])
			Model.summary()
			Model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
			return Model

		def Model_train(Model):
			X, Y = Indexer()
			his = Model.fit(X, Y, batch_size=32, epochs=50)
			return his

		def squence_t_t(list_o_index):
			words = [tokens.get_vocabulary()[let] for let in list_o_index]
			return words

		def create_text():
			Model = Model_create()
			Model_train(Model)

			ttt = "Труд освобождает нелюдей".lower()
			dat = tokens(ttt)
			print(dat)
			dat_paa = pad_sequences([dat], maxlen=maXT)
			print(squence_t_t(dat))
			res = Model.predict(dat_paa)
			print(res, np.argmax(res), sep='\n')

		create_text()

	def less_work_nine_BIRECTIONAL(self):
		N = 1000
		Data = np.array([np.sin(x / 20) for x in range(N)]) + 0.1 * np.random.randn(N)
		plt.plot(Data[:100])
		plt.show()

		fff = 3
		length = fff * 2 + 1
		X = np.array([np.diag(np.hstack((Data[i:i + fff], Data[i + fff + 1:i + length]))) for i in range(N - length)])
		Y = Data[fff:N - fff - 1]
		print(X.shape, Y.shape, sep='\n')

		def model_create():
			Model = Sequential([
				Input((length - 1, length - 1)),
				Bidirectional(GRU(2)),
				Dense(1, activation='linear'),
			])
			Model.compile(loss='mean_squared_error', optimizer=Adam(0.01))
			Model.summary()
			Model.fit(X, Y, batch_size=32, epochs=10)
			return Model

		Model = model_create()

		m = 200
		XX = np.zeros(m)
		XX[:fff] = Data[:fff]
		for i in range(m - fff - 1):
			x = np.diag(np.hstack((XX[i:i + fff], Data[i + fff + 1:i + length])))
			x = np.expand_dims(x, axis=0)
			y = Model.predict(x)
			XX[i + fff + 1] = y
		plt.plot(XX[:m])
		plt.plot(Data[:m])
		plt.show()
	def less_work_ten_Autoencoder(self):
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

		x_train = x_train / 255
		x_test = x_test / 255

		x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
		x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

		hidden_dim = 2
		batch_size = 60

		def dropout_and_Batch(x):
			return Dropout(0.3)(BatchNormalization()(x))

		inp_imm = Input(batch_shape=(batch_size, 28, 28, 1))
		x = Flatten()(inp_imm)
		x = Dense(256, activation='relu')(x)
		x = dropout_and_Batch(x)
		x = Dense(128, activation='relu')(x)
		x = dropout_and_Batch(x)
		z_mmm = Dense(hidden_dim)(x)
		z_logger_v = Dense(hidden_dim)(x)

		def nisee(args):
			global z_mmm, z_logger_v
			z_mmm, z_logger_v = args
			N = (keras.initializers.RandomNormal(mean=0., stddev=1.0)(shape=(batch_size, hidden_dim)))
			return exponential(z_logger_v / 2) * N + z_mmm

		H = Lambda(nisee, output_shape=(hidden_dim,))([z_mmm, z_logger_v])

		inp_dee = Input(shape=(hidden_dim,))
		d = Dense(128, activation='relu')(inp_dee)
		d = dropout_and_Batch(d)
		d = Dense(256, activation='relu')(inp_dee)
		d = dropout_and_Batch(d)
		d = Dense(28 * 28, activation='sigmoid')(d)
		decoded = Reshape((28, 28, 1))(d)

		encoder = keras.Model(inp_imm, H, name='Encoder')
		decoder = keras.Model(inp_dee, decoded, name='Decoder')

		VVV = keras.Model(inp_imm, decoder(encoder(inp_imm)), name='VVV')
		VVV.summary()

		def vVV_loss(x, y):
			x = tf.reshape(x, shape=(batch_size, 28 * 28))
			y = tf.reshape(y, shape=(batch_size, 28 * 28))
			loss = tf.reduce_sum(tf.square(x - y), axis=-1)
			kl_loss = -0.5 * tf.reduce_sum(1 + z_logger_v - tf.square(z_mmm) - exponential(z_logger_v), axis=-1)
			return loss + kl_loss

		VVV.compile(optimizer='adam', loss=vVV_loss)
		VVV.fit(x_train, x_train, epochs=5, batch_size=batch_size, shuffle=True)

		# Начало пользования

		h = encoder.predict(x_test[:6000], batch_size=batch_size)
		plt.scatter(h[:, 0], h[:, 1])
		plt.show()

		n = 5
		totitt = 2 * n + 1
		plt.figure(figsize=(totitt / 2, totitt / 2))

		num = 1
		for i in range(-n, n + 1):
			for j in range(-n, n + 1):
				ax = plt.subplot(totitt, totitt, num)
				num += 1
				imgg = decoder.predict(np.expand_dims([3 * i / n, 3 * j / n], axis=0))
				plt.imshow(imgg.squeeze(), cmap='gray')
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)
		plt.show()
	def less_work_oneten_RASH_Autoencoder(self):
		hidden_dim = 2
		num_class = 10
		batch_size = 100

		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		x_train = x_train / 255
		x_test = x_test / 255

		x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
		x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

		y_train_cat = keras.utils.to_categorical(y_train, num_class)
		y_test_cat = keras.utils.to_categorical(y_test, num_class)

		def Dropouter_and_BN(x):
			return Dropout(0.3)(BatchNormalization()(x))

		inp_imm = Input(shape=(28, 28, 1))
		Fltt = Flatten()(inp_imm)
		LAB = Input(shape=(num_class,))
		x = concatenate([Fltt, LAB])
		x = Dense(256, activation='relu')(x)
		x = Dropouter_and_BN(x)
		x = Dense(128, activation='relu')(x)
		x = Dropouter_and_BN(x)

		Zmather = Dense(hidden_dim)(x)
		ZloggerVar = Dense(hidden_dim)(x)

		def noiserr(args):
			global Zmather, ZloggerVar
			Zmather, ZloggerVar = args
			N = initializers.random_normal(mean=0., stddev=1.0)(shape=(batch_size, hidden_dim))
			return exponential(ZloggerVar / 2) * N + Zmather

		H = Lambda(noiserr, output_shape=(hidden_dim,))([Zmather, ZloggerVar])

		inpt_dec = Input(shape=(hidden_dim,))
		LB_Dec = Input(shape=(num_class,))
		d = concatenate([inpt_dec, LB_Dec])
		d = Dense(128, activation='elu')(d)
		d = Dropouter_and_BN(d)
		d = Dense(256, activation='elu')(d)
		d = Dropouter_and_BN(d)
		d = Dense(28 * 28, activation='sigmoid')(d)
		decoded = Reshape((28, 28, 1))(d)

		encoder = keras.Model([inp_imm, LAB], H, name='encoder')
		Decoder = keras.Model([inpt_dec, LB_Dec], decoded, name='Decoder')
		CONV_VAR_ENCODER = keras.Model([inp_imm, LAB, LB_Dec], Decoder([encoder([inp_imm, LAB]), LB_Dec]), name='Cvae')

		Z_meanger = keras.Model([inp_imm, LAB], Zmather)
		Z_loggerVar = keras.Model([inp_imm, LAB, LB_Dec], Decoder([Z_meanger([inp_imm, LAB]), LB_Dec]),
								  name='TRAt_style')

		def VVV_loss(x, y):
			x = tf.reshape(x, shape=(batch_size, 28 * 28))
			y = tf.reshape(y, shape=(batch_size, 28 * 28))
			loss = tf.reduce_sum(tf.square(x - y), axis=-1)
			KK_loss = -0.5 * tf.reduce_sum(1 + ZloggerVar - tf.square(Zmather) - exponential(ZloggerVar), axis=-1)
			return (loss + KK_loss) / 2 / 28 / 28

		CONV_VAR_ENCODER.compile(optimizer='adam', loss=VVV_loss)

		CONV_VAR_ENCODER.fit([x_train, y_train_cat, y_train_cat], x_train, epochs=5, batch_size=batch_size,
							 shuffle=True)

		LAB = LB_Dec = y_test_cat
		h = encoder.predict([x_test, LAB], batch_size=batch_size)
		plt.scatter(h[:, 0], h[:, 1])
		plt.show()

		n = 4
		total = 2 * n + 1
		input_lbl = np.zeros((1, num_class))
		input_lbl[0, 9] = 1

		plt.figure(figsize=(total, total))

		h = np.zeros((1, hidden_dim))
		num = 1
		for i in range(-n, n + 1):
			for j in range(-n, n + 1):
				ax = plt.subplot(total, total, num)
				num += 1
				h[0, :] = [1 * i / n, 1 * j / n]
				img = Decoder([h, input_lbl])
				plt.imshow(tf.squeeze(img), cmap='gray')
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)
		plt.show()

		def plot_nii(*images):
			images = [x.squeeze() for x in images]
			n = min([x.shape[0] for x in images])
			plt.figure(figsize=(n, len(images)))
			for j in range(n):
				for i in range(len(images)):
					ax = plt.subplot(len(images), n, i * n + j + 1)
					plt.imshow(images[i][j])
					plt.gray()
					ax.get_xaxis().set_visible(False)
					ax.get_yaxis().set_visible(False)
			plt.show()

		ddd = 5
		num = 10
		X = x_train[y_train == ddd][:num]
		lb_1 = np.zeros((num, num_class))
		lb_1[:, ddd] = 1
		plot_nii(X)
		for i in range(num_class):
			lb_2 = np.zeros((num, num_class))
			lb_2[:, i] = 1
			Y = Z_loggerVar.predict([X, lb_1, lb_2], batch_size=num)
			plot_nii(Y)
		plt.show()
	def My_Neuron_test(self):
		def work_1():
			(x_Train, y_Train), (x_test, y_test) = mnist.load_data()

			x_Train = x_Train / 255
			x_test = x_test / 255

			y_train_category = keras.utils.to_categorical(y_Train, 10)
			y_test_category = keras.utils.to_categorical(y_test, 10)

			def first_25num():
				plt.figure(figsize=(10,5))
				for i in range(25):
					plt.subplot(5,5, i + 1)
					plt.xticks([])
					plt.yticks([])
					plt.imshow(x_Train[i], cmap=plt.cm.binary)
				plt.show()
			# first_25num()

			model = keras.Sequential([
				Flatten(input_shape=(28, 28, 1)),
				Dense(2000, activation='relu'),
				Dense(200, activation='relu'),
				Dense(200, activation='relu'),
				Dense(10, activation='softmax')
			])
			model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
			# print(model.summary())
			model.fit(x_Train, y_train_category, batch_size=40, epochs=5, validation_split=0.2)

			Number = 400

			x = np.expand_dims(x_test[Number], axis=0)
			res = model.predict(x)
			print(res)
			print(f"Распознанно: {np.argmax(res)}")

			plt.imshow(x_test[Number], cmap=plt.cm.binary)
			plt.show()

			predon = model.predict(x_test)
			predon = np.argmax(predon, axis=1)

			mas = predon == y_test
			print(mas[:10])
			x_f = x_test[~mas]
			p_f = predon[~mas]
			print(x_f.shape)
			for i in range(100):
				print("Neuron_model_exit: "+str(p_f[i]))
				plt.imshow(x_f[i], cmap=plt.cm.binary)
				plt.show()
		def work_2():
			(x_Train, y_Train), (x_test, y_test) = mnist.load_data()

			x_Train = x_Train / 255
			x_test = x_test / 255

			Y_Tcateg = keras.utils.to_categorical(y_Train, 10)
			Y_teca = keras.utils.to_categorical(y_test, 10)

			Model = keras.Sequential([
				Flatten(input_shape=(28, 28, 1)),
				Dense(50, activation='relu'),
				Dense(10, activation='relu'),
				Dense(10, activation='relu'),
				Dense(10, activation='softmax')
			])
			Model.compile(optimizer='adam',
						  loss='categorical_crossentropy',
						  metrics=['accuracy'])

			Model.fit(x_Train, Y_Tcateg, batch_size=30, epochs=5, validation_split=0.2)

			for el in range(50, 100):
				x = np.expand_dims(x_test[el], axis=0)
				foas = Model.predict(x)
				print(foas)
				print(f"Nime: {np.argmax(foas)}")

				plt.imshow(x_test[el], cmap=plt.cm.binary)
				plt.show()
		def work_3():

			(x_Train, y_Train), (x_test, y_test) = mnist.load_data()

			x_Train = x_Train / 255
			x_test = x_test / 255

			y_Train_cat = keras.utils.to_categorical(y_Train, 10)
			y_test_cat = keras.utils.to_categorical(y_test, 10)

			x_Train = np.expand_dims(x_Train, axis=3)
			x_test = np.expand_dims(x_test, axis=3)

			def Model():
				model = keras.Sequential([
					Conv2D(32, (2, 2), padding="same", activation='relu', input_shape=(28, 28, 1)),
					MaxPooling2D((2, 2), strides=2),
					Conv2D(64, (3, 3), padding='same', activation='relu'),
					Dropout(0.2),
					MaxPooling2D((2, 2), strides=2),
					Flatten(),
					Dense(130, activation='relu'),
					Dropout(0.2),
					Dense(10, activation='softmax')
				])
				model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

				print(model.summary())
				hit = model.fit(x_Train, y_Train_cat, batch_size=30, epochs=10, validation_split=0.2)
				plt.plot(hit.history['loss'])
				plt.plot(hit.history['val_loss'])
				plt.show()
				return model

			def test_two(model):
				eror = model.predict(x_test)
				eror = np.argmax(eror, axis=1)

				maskes = eror == y_test
				print(maskes[:10])
				x_eror = x_test[~maskes]
				f_eror = eror[~maskes]
				print(x_eror.shape)

				for i in range(30):
					print("Значение сети:" + str(f_eror[i]))
					plt.imshow(x_eror[i], cmap=plt.cm.binary)
					plt.show()

			def test_one(model):
				for n in range(10, 1000, 10):
					x = np.expand_dims(x_test[n], axis=0)
					ito = model.predict(x)
					print(ito)
					print(f"Распознанно: {np.argmax(ito)}")
					plt.imshow(x_test[n], cmap=plt.cm.binary)
					plt.show()

			test_two(Model())

		def work_4():
			with open('reccurent/citat.txt', 'r', encoding='utf-8') as f2:
				text = f2.read()
				text = text.replace('\ufeff', '')
				text = re.sub(r'[^А-я]', ' ', text)
				text = text.replace(' г ', '')

			def new_inigma(text):
				maxW = 221700
				tokenizer = TextVectorization(
					max_tokens=maxW,
					standardize='lower_and_strip_punctuation',
					output_mode='int',
					split='whitespace'
				)

				tokenizer.adapt([text])
				data = tokenizer([text])
				res = data[0]
				inP = 3
				N = res.shape[0] - inP
				return maxW, inP, tokenizer, res

			maxW, inpute, tokenizer, res = new_inigma(text)

			def creator_X_Y(ress):
				ress = [x for x in ress if x < 30000]
				ress = np.array(ress)
				N = ress.shape[0] - 3
				X = np.array([ress[i:i + inpute] for i in range(N)])
				Y = to_categorical(ress[inpute:], num_classes=30000)
				# print(X[:10])
				# print(Y[:1])
				# print(X.shape, '\n', Y.shape)
				return X, Y

			def delimiter(res):
				itoger = np.array_split(res, 8)
				return itoger

			def create_model_DANGER():
				kek = delimiter(res)

				Model = Sequential()
				Model.add(Embedding(30000, 256))
				Model.add(SimpleRNN(300, activation='tanh'))
				Model.add(Dropout(0.6))
				Model.add(Dense(30000, activation='softmax'))
				Model.summary()
				Model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
				for ind, el in enumerate(kek):
					print(f'\n\niteration: №{ind}\n\n')
					X, Y = creator_X_Y(el)
					Model.fit(X, Y, batch_size=64, epochs=10)
				Model.save('leon Толстый, Т9 19 века, проглоченная война и мир.h5')
				return Model

			Model = keras.src.saving.legacy.save.load_model('leon Толстый, Т9 19 века, проглоченная война и мир.h5')

			def build_words(texts, str_len=100):
				tee = texts
				data = list(tokenizer([texts])[0])
				for i in range(str_len):
					x = data[i:i + inpute]
					inn = np.expand_dims(x, axis=0)

					pp = Model.predict(inn)
					indxe = pp.argmax(axis=1)[0]
					data.append(indxe)

					tee += " " + tokenizer.get_vocabulary()[indxe].lower()
				return tee

			test = 'Позитив добавляет годы'
			itog = build_words(test)
			itoger = textwrap.fill(itog, 50)

		def work_5():
			pass


		#||||||||||||||||||||||||||||||||||||||||||
		def keyer():
			key = input("My works: (1, 2, 3, 4 ...) \nPress: ")
			return key
		def validat(key):
			if key == "1":
				work_1()
			elif key == "2":
				work_2()
			elif key == "3":
				work_3()
			elif key == "4":
				work_4()
			elif key == '5':
				work_5()
			else:
				print(f"work {key} is nothing")
		validat(keyer())

def main():
	Neuron_keras_lessons()

if __name__ == "__main__":
	main()
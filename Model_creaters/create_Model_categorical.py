import os, logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = 3

from keras.layers import Dense, Embedding, SimpleRNN, Flatten
from keras import Sequential


def create_Model_categorical(intore=None, X=None, Y=None, **kwargs):
	Model = Sequential([
		Embedding(100000, 1000, input_shape=(100,)),
		SimpleRNN(100, activation='relu'),
		Dense(100, activation='relu'),
		Dense(100, activation='relu'),
		Dense(intore, activation='softmax')
	])
	Model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
	Model.summary()

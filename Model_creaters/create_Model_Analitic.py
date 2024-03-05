import os, logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = 3

from keras.layers import Dense,LSTM, Embedding, SimpleRNN
from keras import Sequential



def create_Model_Analitic(intore=None, X=None, Y=None, **kwargs):
	Model = Sequential(
		[Embedding(1000, 300, input_shape=(10,)),
		LSTM(100, activation='tanh'),
		SimpleRNN(1000, activation='relu'),
		Dense(500, activation='relu'),
		Dense(500, activation='relu'),
		Dense(500, activation='relu'),
		Dense(500, activation='relu'),
		Dense(500, activation='relu'),
		Dense(5, activation='softmax')]
	)
	Model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
	Model.summary()
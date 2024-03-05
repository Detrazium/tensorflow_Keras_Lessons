import os, logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = 3

from keras.layers import Dense,LSTM, TextVectorization, Embedding, SimpleRNN, Flatten
from keras import Sequential

def create_ModeL_RNN(Ember, X=None, Y=None, **kwargs):
	Model = Sequential()
	Model.add(Embedding(Ember[0],Ember[1], input_shape=(Ember[2],)))
	Model.add(SimpleRNN(300, activation='tanh'))
	Model.add(Dense(200, activation='softmax'))
	Model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
	Model.summary()
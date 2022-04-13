import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import * #Input, Dense, LeakyReLU, Dropout, Softmax,Activation, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D

def cnn(num_classes=2, input_shape=(101,101,87),dropout=0.1,regularizer=None, initializer='glorot_normal'):
	inputs = Input(input_shape)

	conv1 = Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid', activation='relu')(inputs)
	#conv1 = BatchNormalization(axis=-1,momentum=0.99,epsilon=1e-5)(conv1)
	drop1 = Dropout(dropout)(conv1)

	conv2 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(drop1)
	#conv2 = BatchNormalization(axis=-1,momentum=0.99,epsilon=1e-5)(conv2)
	drop2 = Dropout(dropout)(conv2)

	pool1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='valid')(drop2)

	conv3 = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(pool1)
	#conv3 = BatchNormalization(axis=-1,momentum=0.99,epsilon=1e-5)(conv3)
	drop3 = Dropout(dropout)(conv3)

	conv4 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(drop3)
	#conv4 = BatchNormalization(axis=-1,momentum=0.99,epsilon=1e-5)(conv4)
	drop4 = Dropout(dropout)(conv4)

	pool2 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='valid')(drop4)

	conv5 = Conv2D(filters=768, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(pool2)
	#conv5 = BatchNormalization(axis=-1,momentum=0.99,epsilon=1e-5)(conv5)
	drop5 = Dropout(dropout)(conv5)

	conv6 = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(drop5)
	#conv6 = BatchNormalization(axis=-1,momentum=0.99,epsilon=1e-5)(conv6)
	drop6 = Dropout(dropout)(conv6)

	conv7 = Conv2D(filters=1536, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(drop6)
	#conv7 = BatchNormalization(axis=-1,momentum=0.99,epsilon=1e-5)(conv7)
	drop7 = Dropout(dropout)(conv7)

	conv8 = Conv2D(filters=2048, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(drop7)
	#conv8 = BatchNormalization(axis=-1,momentum=0.99,epsilon=1e-5)(conv8)
	drop8 = Dropout(dropout)(conv8)

	net = GlobalAveragePooling2D()(drop8)

	# 1st Fully Connected Layer
	net = Dense(2048,activation='relu')(net)
	net = Dropout(0.2)(net)

	#net = Dense(128, activation='relu')(net)
	#net = Dropout(0.2)(net)

	net = Dense(1,activation='sigmoid')(net)
	model = Model(inputs = inputs, outputs = net)

	model.summary()
	return model

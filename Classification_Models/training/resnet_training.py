import json
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Conv2D, Activation, Dropout, add, Input, Dense, Flatten
from keras.models import Model
from keras.utils.np_utils import to_categorical


class DataGenerator(keras.utils.Sequence):
    def __init__(self, dataframe, vectors_of_words, batch_size=128, n_classes=10, shuffle=True):
        self.dataframe = dataframe
        self.indexes = np.arange(self.dataframe.__len__())
        self.n_classes = n_classes
        self.labels = to_categorical(self.dataframe['class'], self.n_classes)
        self.vectors_of_words = vectors_of_words
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(self.dataframe.__len__() / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indices of the batch
        indices = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indices)

        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indices):
        """
        Generates data containing batch_size samples
        """
        # Initialization
        X = []
        y = []

        # Generate data
        for i, idx in enumerate(indices):
            input_indices = json.loads(self.dataframe.iloc[idx, 1])
            vec = self.vectors_of_words[input_indices]
            if vec.__len__() < 100:
                num = 100 - vec.__len__()
                vec = np.append(vec, np.zeros((num, 100)), axis=0)
            elif vec.__len__() > 100:
                vec = vec[:100]
            X.append(vec)
            y.append(self.labels[idx])
        X = np.array(X)
        X = np.reshape(X, [X.shape[0], 100, 100, 1])
        y = np.array(y)
        return X, y


def bn_relu(layer, dropout=0, **params):
    layer = BatchNormalization()(layer)
    layer = Activation(params['conv_activation'])(layer)

    if dropout > 0:
        layer = Dropout(dropout)(layer)
    return layer


def res_block(input, filters, stride, dim_up=False):
    if dim_up:
        shortcut = Conv2D(filters=filters,
                          kernel_size=[1, 1],
                          use_bias=False)(input)
    else:
        shortcut = input

    layer = Conv2D(filters=filters,
                   kernel_size=[1, stride],
                   kernel_initializer='random_uniform',
                   padding='same')(input)
    layer = bn_relu(layer, conv_activation='relu')
    layer = Conv2D(filters=filters,
                   kernel_size=[1, stride],
                   kernel_initializer='random_uniform',
                   padding='same')(layer)
    layer = add([layer, shortcut])
    layer = bn_relu(layer, conv_activation='relu')
    return layer


def network(num):
    input = Input(shape=[100, 100, 1])
    layer = Conv2D(filters=64, kernel_size=[1, 100], strides=[1, 100], kernel_initializer='random_uniform')(input)
    layer = bn_relu(layer, conv_activation='relu')
    layer = res_block(layer, 64, stride=2)
    layer = res_block(layer, 128, stride=4, dim_up=True)
    layer = res_block(layer, 256, stride=8, dim_up=True)
    layer = res_block(layer, 512, stride=16, dim_up=True)
    layer = Flatten()(layer)
    output = Dense(num, activation='softmax')(layer)
    model = Model(inputs=[input], outputs=[output])
    model.summary()
    return model


kinds_of_label = 73
model = network(kinds_of_label)

word2vec_model = Word2Vec.load('../history_word_embedding.CBOW')
vectors = word2vec_model.wv.vectors


def custom_loss(y_true, y_pred):
    return keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.1)


adam = optimizers.adam(lr=0.001)
model.compile(loss=custom_loss,
              optimizer=adam,
              metrics=['accuracy'])

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

kinds_of_label = 73
training_batch_size = 64
test_batch_size = 1

train_and_eval_set = pd.read_csv('../datasets/train_set_0.csv')
test_set = pd.read_csv('../datasets/test_set_0.csv')

train_set, eval_set = train_test_split(train_and_eval_set, test_size=0.2)
training_generator = DataGenerator(dataframe=train_set, vectors_of_words=vectors,
                                   batch_size=training_batch_size, n_classes=kinds_of_label)
validation_generator = DataGenerator(dataframe=eval_set, vectors_of_words=vectors,
                                     batch_size=training_batch_size, n_classes=kinds_of_label)
test_generator = DataGenerator(dataframe=test_set, vectors_of_words=vectors,
                               batch_size=test_batch_size, n_classes=kinds_of_label, shuffle=False)

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=20,
                    callbacks=[monitor])

score = model.evaluate_generator(generator=test_generator)
predictions = model.predict_generator(test_generator)
print('test loss:', score[0], '; test accuracy:', score[1])
model.save('models/RCNN_lr001_withLS01.h5')

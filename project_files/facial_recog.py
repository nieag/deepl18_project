from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil


def load_landmarks(path):
    with open(path + '/training.txt') as f:
        lines = f.readlines()
        landmarks = [np.asfarray(line.split()[1:11]) for line in lines]
    train_landmarks = np.asarray(landmarks)

    with open(path + '/testing.txt') as f:
        lines = f.readlines()
        landmarks = [np.asfarray(line.split()[1:11]) for line in lines]
    valid_landmarks = np.asarray(landmarks)

    return train_landmarks, valid_landmarks

def split_data(path, train_amount=0.8):
    train_dir = path + '/training_data/training_data';
    valid_dir = path + '/validation_data/validation_data';

    i = 0
    with open(path + '/temp_small.txt') as f:
        lines = f.readlines()
        n_samples = len(lines)
        print(int(train_amount*n_samples))
        for line in lines:
            current = line.split()[0]
            source = path + '/' + str(current).replace('\\', '/')
            print(source)
            if i < int(train_amount*n_samples):
                shutil.copy(source, train_dir)
            else:
                shutil.copy(source, valid_dir)
            i += 1

    print('Copied {} files'.format(i))

def generate_bottle_neck(path, batch_size):
    model = InceptionV3(include_top=False, weights='imagenet')
    datagen = ImageDataGenerator()


    train_gen = datagen.flow_from_directory(
                path+'/training_data',
                class_mode=None,
                batch_size=batch_size,
                shuffle=False)

    bottle_feat_train = model.predict_generator(train_gen)

    np.save(path + '/bottleneck_features_train.npy', bottle_feat_train)

    valid_gen = datagen.flow_from_directory(
                path+'/validation_data',
                target_size=(150, 150),
                class_mode=None,
                batch_size=batch_size,
                shuffle=False)

    bottle_feat_valid = model.predict_generator(valid_gen)
    np.save(path + '/bottleneck_features_valid.npy', bottle_feat_valid)


def train_top_regression(path):
    train_data = np.load(path + '/bottleneck_features_train.npy')
    valid_data = np.load(path + '/bottleneck_features_valid.npy')
    

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(10, activation=None))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(valid_data, valid_labels))
    model.save_weights(path)

if __name__ == '__main__':
    path = '/home/niels/Documents/deepl18_project/MTFL'

    # split_data(path)
    # generate_bottle_neck(path, 50)

    # train_data = np.load(path+'/bottleneck_features_train.npy')
    # valid_data = np.load(path+'/bottleneck_features_valid.npy')
    # print(train_data)
    # print(valid_data)

    train_top_regression(path)

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

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
    train_dir = path;
    valid_dir = temp;

    i = 1
    with open(path + '/training.txt') as f:
        lines = f.readlines()
        n_samples = len(lines)
        for line in lines:
            current = line.split()[0]
            source = path + str(current).replace('\\', '/')
            if i != train_amount*n_samples:
                shutil.copy(source, train_dir)
            else:
                shutil.copy(source, valid_dir)
            i += 1




def generate_bottle_neck(path, batch_size):

    model = InceptionV3(include_top=False, weights='imagenet')
    datagen = ImageDataGenerator()

    train_gen = datagen.flow_from_directory(
                path,
                target_size=(150, 150),
                class_mode=None,
                batch_size=batch_size,
                shuffle=False)

    bottle_feat_train = model.predict_generator(train_gen, n_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'w'), bottle_feat_train)

    valid_gen = datagen.flow_from_directory(
                path,
                target_size=(150, 150),
                class_mode=None,
                batch_size=batch_size,
                shuffle=False)

    bottle_feat_valid = model.predict_generator(valid_gen, n_train_samples // batch_size)
    np.save(open('bottleneck_features_valid.npy', 'w'), bottle_feat_valid)


def train_top_regression(train_path, valid_path):
    train_data = np.load(open(train_path))

    valid_data = np.load(open(valid_path))

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
-
    train_landmark, valid_landmark = load_landmarks(path)

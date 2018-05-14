from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, Input, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import Adam
from keras import regularizers
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import shutil
import glob
import csv
from keras.datasets import boston_housing

import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle


def inception_regression(X, y, validation_data, epochs):
    datagen = ImageDataGenerator()
    # datagen.fit(X)

    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    # x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.5)(x)
    predictions = Dense(10, activation=None)(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    print(model.summary())
    # Freeze the inception model
    for layer in base_model.layers:
        layer.trainable = False
    # optimizer = Adam(lr=0.01)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    # print(model.summary())
    # Callbacks
    early_stop = EarlyStopping(patience=20)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=True)

    model.fit_generator(datagen.flow(X, y, batch_size=32, shuffle=True),
                                     epochs=epochs,
                                     validation_data=validation_data,
                                     callbacks=[tensorboard, early_stop])

    return model

def split_data(path):
    train_dir = path + '/training_data/training_data';
    i = 0
    with open(path + '/training.txt') as f:
    # with open(path + '/temp.txt') as f:
        lines = f.readlines()
        for line in lines:
            current = line.strip('\n').split()[0]
            source = path + '/' + str(current).replace('\\', '/')
            print(source)
            shutil.copy(source, train_dir)
            i += 1
    print('Copied {} files'.format(i))

    i=0
    test_dir = path + '/testing_data/testing_data'
    with open(path + '/testing.txt') as f:
    # with open(path + '/temp_test.txt') as f:
        lines = f.readlines()
        for line in lines:
            current = line.strip('\n').split()[0]
            source = path + '/' + str(current).replace('\\', '/')
            print(source)
            shutil.copy(source, test_dir)
            i += 1
    print('Copied {} files'.format(i))

def import_images(path, text_path):
    image_list = []
    size = 150
    landmarks = []
    with open(text_path) as f:
        lines = f.readlines()
        for line in lines:
            current = line.strip('\n').split()[0]
            im_path = path + '/' + str(current).replace('\\', '/')
            im=Image.open(im_path)
            if(im.size[0] == im.size[1] and np.max(im) != 0 and im.mode=='RGB'):
                ratio = size/im.size[0]
                im.thumbnail((size, size))
                im = im/np.max(im)
                landmark = np.asfarray(line.strip('\n').split()[1:11])*ratio
                landmark = np.divide((landmark-size/2),(size/2))
                image_list.append(im)
                landmarks.append(landmark)
    image_list = np.array(image_list)
    landmarks = np.array(landmarks)
    image_list = image_list.astype(np.float32)
    landmakrs = landmarks.astype(np.float32)
    return image_list, landmarks

def plot_sample(img, y):
    plt.figure()
    plt.imshow(img)
    plt.scatter(y[:5] * 75 + 75, y[5:] * 75 + 75)

def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

if __name__ == '__main__':
    path = '/home/niels/Documents/deepl18_project/MTFL'
    im = '/home/niels/Documents/deepl18_project/MTFL/AFLW/0001-image20056.jpg'
    train_land_path = '/home/niels/Documents/deepl18_project/MTFL/temp.txt'
    test_land_path = '/home/niels/Documents/deepl18_project/MTFL/temp_test.txt'

    FTRAIN = '/home/niels/Downloads/training.csv'
    FTEST = '/home/niels/Downloads/test.csv'

    """GCP paths"""
    # path = '/home/niels_agerskov/deepl18_project/MTFL'
    # train_land_path = '/home/niels_agerskov/deepl18_project/MTFL/training.txt'
    # test_land_path = '/home/niels_agerskov/deepl18_project/MTFL/testing.txt'

    # trainX, trainy = load()
    # testX, testy = load(test=True)
    # print(trainX.shape)
    # print(x_train.shape)
    # print(y_train)
    # split_data(path)
    train_images, train_landmarks = import_images(path, train_land_path)
    # print(np.max(train_images))
    # print(np.min(train_images))
    # print(np.max(train_landmarks))
    # print(np.min(train_landmarks))
    test_images, test_landmarks = import_images(path, test_land_path)
    # print(test_landmarks.shape)
    # print(test_landmarks)
    model = inception_regression(train_images, train_landmarks, (test_images, test_landmarks), 10)
    # model.save(path+'/test_model.h5')
    # model = load_model(path+'/test_model.h5')
    # prediction = model.predict(test_images)
    # np.save('prediction.npy', prediction)
    # # print(prediction[0])
    # # print(test_landmarks[0])
    # # test = [1, 1, 1]
    # # test = np.array(test)
    # # np.save('test.npy', test)
    # predictions = np.load('prediction.npy')
    # for i in range(10):
    #     print(predictions[i]*75 + 75)
    #     plot_sample(test_images[i], predictions[i])
    # plt.show()

    # test_regression(train_images, train_landmarks, (test_images, test_landmarks), 10)

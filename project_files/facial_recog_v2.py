from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, Input
from keras.callbacks import TensorBoard, EarlyStopping
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import shutil
import glob
import csv
import pandas as pd



def inception_regression(X, y, validation_data, epochs):
    datagen = ImageDataGenerator(samplewise_center=True,
                                samplewise_std_normalization=True,
                                # rotation_range=20,
                                # width_shift_range=0.2,
                                # height_shift_range=0.2,
                                horizontal_flip=False)
    # datagen.fit(X)

    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(150, 150, 3)))
    x = base_model.output
    # x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    predictions = Dense(10, activation='linear')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # Freeze the inception model
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    for layer in base_model.layers:
        print(layer.trainable)

    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'acc'])
    # print(model.summary())
    # Callbacks
    early_stop = EarlyStopping(patience=20)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=True)
    model.fit_generator(datagen.flow(X, y, batch_size=32, shuffle=True),
                                     validation_data=validation_data,
                                     epochs=epochs,
                                     callbacks=[tensorboard, early_stop])
    # model.fit(X, y, validation_data=validation_data, epochs=epochs, callbacks=[tensorboard])
    return model


def split_data(path):
    train_dir = path + '/training_data/training_data';
    i = 0
    # with open(path + '/training.txt') as f:
    with open(path + '/temp.txt') as f:
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
    # with open(path + '/testing.txt') as f:
    with open(path + '/temp_test.txt') as f:
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
            if(im.size[0] == im.size[1] and im.mode=='RGB'):
                ratio = size/im.size[0]
                im.thumbnail((size, size))
                im = im/np.max(im)
                im = im.astype(np.float32)
                landmark = np.asfarray(line.strip('\n').split()[1:11])*ratio
                landmark = (landmark-75)/75
                landmark = landmark.astype(np.float32)
                image_list.append(im)
                landmarks.append(landmark)
    image_list = np.array(image_list)
    landmarks = np.array(landmarks)

    return image_list, landmarks

def plot_sample(img, y):
    plt.figure()
    plt.imshow(img)
    plt.scatter(y[:5] * 75 + 75, y[5:] * 75 + 75)

if __name__ == '__main__':
    path = '/home/niels/Documents/deepl18_project/MTFL'
    im = '/home/niels/Documents/deepl18_project/MTFL/AFLW/0001-image20056.jpg'
    train_land_path = '/home/niels/Documents/deepl18_project/MTFL/temp.txt'
    test_land_path = '/home/niels/Documents/deepl18_project/MTFL/temp_test.txt'

    """GCP paths"""
    # path = '/home/niels_agerskov/deepl18_project/MTFL'
    # train_land_path = '/home/niels_agerskov/deepl18_project/MTFL/training.txt'
    # test_land_path = '/home/niels_agerskov/deepl18_project/MTFL/testing.txt'

    # split_data(path)
    train_images, train_landmarks = import_images(path, train_land_path)
    # print(np.max(train_images))
    # print(np.min(train_images))
    # print(np.max(train_landmarks))
    # print(np.min(train_landmarks))
    test_images, test_landmarks = import_images(path, test_land_path)
    model = inception_regression(train_images, train_landmarks, (test_images, test_landmarks), 20)
    # model.save(path+'/test_model.h5')
    # model = load_model(path+'/test_model.h5')
    # prediction = model.predict(test_images)
    # print(prediction[0])
    # print(test_landmarks[0])
    # test = [1, 1, 1]
    # test = np.array(test)
    # np.save('test.npy', test)
    # predictions = np.load('prediction.npy')
    # for i in range(10):
    #     print(predictions[i]*75 + 75)
    #     plot_sample(test_images[i], predictions[i])
    # plt.show()

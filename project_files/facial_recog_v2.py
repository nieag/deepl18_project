from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Flatten, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import glob

def inception_regression(X, y, validation_data, epochs):
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    # x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(10, activation='linear')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, validation_data=validation_data, epochs=epochs, batch_size=32)
    return model

def split_data(path):
    train_dir = path + '/training_data/training_data';

    i = 0
    with open(path + '/training.txt') as f:
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
    size = 100
    landmarks = []
    with open(text_path) as f:
        lines = f.readlines()
        for line in lines:
            current = line.strip('\n').split()[0]
            im_path = path + '/' + str(current).replace('\\', '/')
            im=Image.open(im_path)
            if(im.size[0] == im.size[1] and im.mode=='RGB'):
                ratio = size/im.size[0]
                # ratio = 1
                im.thumbnail((size, size))
                im = im/np.max(im)
                im = im.astype(np.float32)
                landmark = np.asfarray(line.strip('\n').split()[1:11])*ratio
                landmark = (landmark-50)/50
                landmark = landmark.astype(np.float32)
                # landmark = (landmark - np.mean(landmark))/np.std(landmark)
                image_list.append(im)
                landmarks.append(landmark)
    image_list = np.array(image_list)
    landmarks = np.array(landmarks)

    return image_list, landmarks

if __name__ == '__main__':
    path = '/home/niels/Documents/deepl18_project/MTFL'
    im = '/home/niels/Documents/deepl18_project/MTFL/AFLW/0001-image20056.jpg'
    train_land_path = '/home/niels/Documents/deepl18_project/MTFL/temp.txt'
    test_land_path = '/home/niels/Documents/deepl18_project/MTFL/temp_test.txt'

    """GCP paths"""
    path = '/home/niels/Documents/deepl18_project/MTFL'
    train_land_path = '/home/niels/Documents/deepl18_project/MTFL/training.txt'
    test_land_path = '/home/niels/Documents/deepl18_project/MTFL/testing.txt'

    split_data(path)
    train_images, train_landmarks = import_images(path, train_land_path)
    # print(np.max(train_images))
    # print(np.min(train_images))
    # print(np.max(train_landmarks))
    # print(np.min(train_landmarks))
    test_images, test_landmarks = import_images(path, test_land_path)
    print(test_images.shape)
    # model = inception_regression(train_images, train_landmarks, (test_images, test_landmarks), 10)
    # model.save(path+'/test_model.h5')
    # model = load_model(path+'/test_model.h5')
    # prediction = model.predict(test_images)
    # print(prediction[0])
    # print(test_landmarks[0])
    # plt.figure()
    # plt.imshow(test_images[0])
    # plt.scatter(test_landmarks[0][:5], test_landmarks[0][5:])
    # plt.show()

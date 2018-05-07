from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

def visualise_sample(img, landmarks):
    img = load_img(img)
    # lines = open(path + '/temp.txt')
    # landmarks = [np.asfarray(line.strip('\n').split()[1:11]) for line in lines]
    # landmarks = np.asarray(landmarks)

    # test = landmarks[0]
    landmark = landmarks[0]
    plt.figure()
    plt.imshow(img)
    plt.scatter(landmark[:5], landmark[5:])
    plt.show()

def get_landmarks(path, train_amount=1):
    with open(path + '/temp.txt') as f:
        lines = f.readlines()
        landmarks = [np.asfarray(line.strip('\n').split()[1:11]) for line in lines]
        landmarks = np.asarray(landmarks)

    n_samples = len(landmarks)
    split = np.split(landmarks, [int(n_samples*train_amount)])
    train_landmarks = split[0]
    valid_landmarks = split[1]

    with open(path + '/temp_test.txt') as f:
        lines = f.readlines()
        landmarks = [np.asfarray(line.strip('\n').split()[1:11]) for line in lines]
        landmarks = np.asarray(landmarks)
        testing_landmarks = landmarks

    return train_landmarks, valid_landmarks, testing_landmarks

def split_data(path, train_amount=1):
    train_dir = path + '/training_data/training_data';
    valid_dir = path + '/validation_data/validation_data';

    i = 0
    with open(path + '/temp.txt') as f:
        lines = f.readlines()
        # lines = lines[:-1]
        n_samples = len(lines)
        print(int(train_amount*n_samples))
        for line in lines:
            current = line.strip('\n').split()[0]
            source = path + '/' + str(current).replace('\\', '/')
            print(source)
            if i < int(train_amount*n_samples):
                shutil.copy(source, train_dir)
            else:
                shutil.copy(source, valid_dir)
            i += 1
    print('Copied {} files'.format(i))

    i=0
    test_dir = path + '/testing_data/testing_data'
    with open(path + '/temp_test.txt') as f:
        lines = f.readlines()
        # lines = lines[:-1]
        n_samples = len(lines)
        print(int(train_amount*n_samples))
        for line in lines:
            current = line.strip('\n').split()[0]
            source = path + '/' + str(current).replace('\\', '/')
            print(source)
            shutil.copy(source, test_dir)
            i += 1
    print('Copied {} files'.format(i))

def generate_bottle_neck(path, batch_size):
    model = InceptionV3(include_top=False, weights='imagenet')
    datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, rescale=1./255, )

    train_gen = datagen.flow_from_directory(
                path+'/training_data',
                target_size=(100, 100),
                class_mode=None,
                batch_size=batch_size,
                shuffle=False)

    bottle_feat_train = model.predict_generator(train_gen)

    np.save(path + '/bottleneck_features_train.npy', bottle_feat_train)

    # valid_gen = datagen.flow_from_directory(
    #             path+'/validation_data',
    #             class_mode=None,
    #             batch_size=batch_size,
    #             shuffle=False)
    #
    # bottle_feat_valid = model.predict_generator(valid_gen)
    # np.save(path + '/bottleneck_features_valid.npy', bottle_feat_valid)

    test_gen = datagen.flow_from_directory(
                path+'/testing_data',
                target_size=(100,100),
                class_mode=None,
                batch_size=batch_size,
                shuffle=False)
    bottle_feat_test = model.predict_generator(test_gen)

    np.save(path + '/bottleneck_features_test.npy', bottle_feat_test)

def train_top_regression(path, epochs, batch_size):
    start = 0.03
    stop = 0.001

    train_data = np.load(path + '/bottleneck_features_train.npy')
    test_data = np.load(path + '/bottleneck_features_test.npy')
    train_land, valid_land, test_land = get_landmarks(path)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    # model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(10, activation='linear'))
    # adam = Adam(lr=0.1'
    model.compile(loss='mse', optimizer='rmsprop')

    # sgd = SGD(lr=start, momentum=0.9, nesterov=True)
    # model.compile(loss='mean_squared_error', optimizer=sgd)
    # learning_rates = np.linspace(start, stop, epochs)
    # change_lr = LearningRateScheduler(lambda epoch: float(learning_rates[epoch]))
    #
    # hist = model.fit(train_data, train_land,
    #                  nb_epoch=epochs,
    #                  validation_data=(test_data, test_land),
    #                  callbacks=[change_lr])

    hist = model.fit(train_data, train_land,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(test_data, test_land))
    # hist.save(path+'/test_model.h5')
    print(hist.history)
    plt.plot(hist.history['val_loss'])
    plt.show()
if __name__ == '__main__':
    path = '/home/niels/Documents/deepl18_project/MTFL'
    im = '/home/niels/Documents/deepl18_project/MTFL/AFLW/0001-image20056.jpg'

    # split_data(path)
    generate_bottle_neck(path, 32)
    train_top_regression(path, 50, 32)
    # model = load_model(path+'/test_model.h5')
    # test_data = np.load(path + '/bottleneck_features_test.npy')
    # # # print(test_data)
    # prediction = model.predict(test_data)
    # print(prediction[0])
    # # visualise_sample(im, prediction)

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Flatten, GlobalAveragePooling2D, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import shutil
import glob
from keras import backend as K
from keras.optimizers import SGD



from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def inception_regression(X, y, validation_data, epochs, batch):
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(150,150,3))


    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
   # Dropout(0.5)
    # let's add a fully-connected layer
    x = Dense(2048, kernel_initializer='normal', activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(1, kernel_initializer='normal', activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',  metrics=['accuracy'])

    # train the model on the new data for a few epochs
    model.fit(X, y, validation_data=validation_data, epochs=epochs, batch_size=batch)


    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    #for i, layer in enumerate(base_model.layers):
     #   print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    #for layer in base_model.layers[-4:]:
    #    layer.trainable = True
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True
    # we need to recompile the model for these modifications to take effect
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy',metrics=['accuracy'])



    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers

    model.fit(X, y, validation_data=validation_data, epochs=epochs, batch_size=batch)

   # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    #results = cross_val_score(model, X, y, cv=kfold)
    #print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    return model

def import_images(path, text_path,size):
    image_list = []
    landmarks = []
    with open(text_path) as f:
        lines = f.readlines()
        for line in lines:
            current = line.strip('\n').split()[0]
            im_path = path + '/' + str(current).replace('\\', '/')
            im=Image.open(im_path)
            if(im.size[0] == im.size[1] and im.mode=='RGB'):

                #im.thumbnail((size, size))
                im = im.resize((size, size),)
              #  im = im/np.max(im)
                im = np.array(im) #im.astype(np.float32)
                landmark = np.asfarray(line.strip('\n').split()[12])
                landmark = landmark.astype(int)
                # landmark = (landmark - np.mean(landmark))/np.std(landmark)
                image_list.append(im)
                landmarks.append(landmark)



    encoder = LabelEncoder()
    encoder.fit(landmarks)
    encoded_Y = encoder.transform(landmarks)
    image_list = np.array(image_list)
    print(image_list.shape)
    landmarks = np.array(encoded_Y)

    return image_list, landmarks

if __name__ == '__main__':


    train_land_path = 'C:/Users/samy_/OneDrive/Documents/DeepLearningProject/MTFL/tmp_training.txt'
    test_land_path = 'C:/Users/samy_/OneDrive/Documents/DeepLearningProject/MTFL/tmp_testing.txt'

    size_of_images = 150
    batch_size = 100
    epochs = 3
    """GCP paths"""
    path = 'C:/Users/samy_/OneDrive/Documents/DeepLearningProject/MTFL'
   # train_land_path = 'C:/Users/samy_/OneDrive/Documents/DeepLearningProject/MTFL/training.txt'
    #test_land_path = 'C:/Users/samy_/OneDrive/Documents/DeepLearningProject/MTFL/testing.txt'

    train_images, train_landmarks = import_images(path, train_land_path, size_of_images)

    # print(np.max(train_images))
    # print(np.min(train_images))
    # print(np.max(train_landmarks))
    # print(np.min(train_landmarks))
    test_images, test_landmarks = import_images(path, test_land_path, size_of_images)
    print(test_landmarks)
    #print(test_images.shape)

"""""
    model = inception_regression(train_images, train_landmarks, (test_images, test_landmarks), epochs, batch_size)
    model.save(path+'/test_model.h5')
    model = load_model(path+'/test_model.h5')
    prediction = model.predict(test_images)
    print(prediction[0])
    print(test_landmarks[0])
    plt.figure()
    plt.imshow(test_images[0])
    plt.show()
"""""


from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from skimage.segmentation import mark_boundaries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from skimage.exposure import rescale_intensity
from keras.callbacks import History
from skimage import io
import matplotlib.pyplot as plt
import tensorflow as tf

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
ds = 1
img_rows = int(256/ds)
img_cols = int(256/ds)
batch_size=2
epochs=40
smooth = 1.
#We divide here the number of rows and columns by two because we undersample our data (We take one pixel over two)

def window(imgs, window_lower, window_upper):
    window_img = imgs.copy()
    window_img[window_img < window_lower] = window_lower
    window_img[window_img > window_upper] = window_upper
    return window_img

def load_train_data():
    imgs_train = np.load('E:/CT_scans/Out/imgs_train.npy')
    masks_train = np.load('E:/CT_scans/Out/masks_train.npy')
    return imgs_train, masks_train

def load_test_data():
    imgs_test = np.load('E:/CT_scans/Out/imgs_test.npy')
    masks_test = np.load('E:/CT_scans/Out/masks_test.npy')
    return imgs_test, masks_test

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

#The functions return our metric and loss

def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    up7 = Dropout(0.5)(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    up8 = Dropout(0.5)(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    up9 = Dropout(0.5)(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[dice_coef])

    return model

#The different layers in our neural network model (including convolutions, maxpooling and upsampling)

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols))
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

#We adapt here our dataset samples dimension so that we can feed it to our network

def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = window(imgs_train, -300, 200)
    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    #imgs_train = imgs_train.astype('float32')
    # mean = np.mean(imgs_train)  # mean for data centering
    # std = np.std(imgs_train)  # std for data normalization
    #
    # imgs_train -= mean
    # imgs_train /= std
    #Normalization of the train set

    #imgs_mask_train = imgs_mask_train.astype('float32')

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    #Saving the weights and the loss of the best predictions we obtained

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    history=model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=False,
              validation_split=0.3,
              callbacks=[model_checkpoint])

    print('-'*30)
    print('Saving model...')
    print('-'*30)
    model.save("full_model.h5")
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_maskt = load_test_data()
    imgs_test = window(imgs_test, -300, 200)
    imgs_test = preprocess(imgs_test)
    imgs_maskt = preprocess(imgs_maskt)
    # imgs_test = imgs_train
    # imgs_maskt = imgs_mask_train
    #imgs_test = imgs_test.astype('float32')
    # imgs_test -= mean
    # imgs_test /= std
    #Normalization of the test set

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    #model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1, batch_size=batch_size)
    print(sum(sum(sum(imgs_mask_test))))

    print('-'*30)
    print('Evaluating masks on test data...')
    print('-'*30)
    results = model.evaluate(imgs_test, imgs_maskt, batch_size=batch_size)
    print(results)

    np.save('imgs_mask_test.npy', imgs_mask_test)
    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)

    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    for k in range(len(imgs_mask_test)):
        a=rescale_intensity(imgs_test[k][:,:,0],out_range=(-1,1))
        #a=imgs_test[k][:,:,0]
        b=(imgs_mask_test[k][:,:,0]).astype('uint8')
        #b=imgs_mask_test[k][:,:,0]
        io.imsave(os.path.join(pred_dir, str(k) + '_pred.png'),mark_boundaries(a,b))
    #Saving our predictions in the directory 'preds'
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('Model dice coeff')
    plt.ylabel('Dice coeff')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    #plotting our dice coeff results in function of the number of epochs

def test_only(img_type = 'test'):
    import tensorflow as tf
    print('Loading and preprocessing test data...')
    print('-'*30)

    if img_type == 'test':
        imgs_test, imgs_maskt = load_test_data()
        imgs_test = window(imgs_test, -300, 200)
        imgs_test = preprocess(imgs_test)
        imgs_maskt = preprocess(imgs_maskt)
        cur_img = imgs_test
        cur_mask = imgs_maskt
    if img_type == 'train':
        imgs_train, imgs_mask_train = load_train_data()
        imgs_train = window(imgs_train, -300, 200)
        imgs_train = preprocess(imgs_train)
        imgs_mask_train = preprocess(imgs_mask_train)
        cur_img = imgs_train
        cur_mask = imgs_mask_train

    model = tf.keras.models.load_model("full_model.h5",custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef':dice_coef})
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(cur_img, verbose=1, batch_size=batch_size)

    #print(sum(sum(sum(imgs_mask_test))))
    print(sum(sum(sum(imgs_mask_test))))

    #binarize
    imgs_mask_test[imgs_mask_test > 0.4] = 1
    imgs_mask_test[imgs_mask_test <= 0.4] = 0

    np.save('imgs_mask_test_binary.npy', imgs_mask_test)
    #np.save('imgs_mask_train.npy', imgs_mask_train)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)

    print('-'*30)
    print('Evaluating masks on test data...')
    print('-'*30)
    # results = model.evaluate(imgs_test, imgs_maskt, batch_size=batch_size)
    # print(results)

    # pred_dir = 'preds'
    # if not os.path.exists(pred_dir):
    #     os.mkdir(pred_dir)
    # for k in range(len(imgs_mask_test)):
    #     a=rescale_intensity(imgs_test[k][:,:,0],out_range=(-1,1))
    #     #a=imgs_test[k][:,:,0]
    #     b=(imgs_mask_test[k][:,:,0]).astype('uint8')
    #     #b=imgs_mask_test[k][:,:,0]
    #     io.imsave(os.path.join(pred_dir, str(k) + '_pred.png'),mark_boundaries(a,b))
    pred_dir_train = 'preds_train'
    if not os.path.exists(pred_dir_train):
        os.mkdir(pred_dir_train)
    for k in range(len(imgs_mask_test)):
        a=rescale_intensity(cur_img[k][:,:,0],out_range=(-1,1))
        #a=imgs_test[k][:,:,0]
        b=(imgs_mask_test[k][:,:,0]).astype('uint8')
        #b=imgs_mask_test[k][:,:,0]
        io.imsave(os.path.join(pred_dir_train, str(k) + '_pred_train.png'),mark_boundaries(a,b))


if __name__ == '__main__':
    train_and_predict()
    #test_only(img_type = 'train')
    #conda activate tf USE THIS COMMAND TO ACTIVATE TENSORFLOW

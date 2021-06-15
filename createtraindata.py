import os
import numpy as np
#import nibabel

#ADAPTED FROM https://towardsdatascience.com/medical-images-segmentation-using-keras-7dc3be5a8524

data_path = 'E:/CT_scans/Out'
train_data_path = 'E:/CT_scans/Out/Train'
mask_data_path = 'E:/CT_scans/Out/Train_mask'
test_data_path = 'E:/CT_scans/Out/Test'
test_mask_data_path = 'E:/CT_scans/Out/Test_mask'
#we will undersample our training 2D images later (for memory and speed)
ds=1
image_rows = int(256/ds)
image_cols = int(256/ds)

def create_train_data():
    print('-'*30)
    print('Creating training data...')
    print('-'*30)
    #file names corresponding to training images
    training_images = os.listdir('E:/CT_scans/Out/Train')
    #file names corresponding to training masks
    training_masks = os.listdir('E:/CT_scans/Out/Train_mask')
    #training images
    imgs_train = []
    #training masks (corresponding to the liver)
    masks_train = []

    for aorta, orig in zip(training_masks, training_images):
        #we load 3D training mask (shape=(X,256,256))
        training_mask = np.load(os.path.join(mask_data_path, aorta))
        #we load 3D training image
        training_image = np.load(os.path.join(train_data_path, orig))

        for k in range(training_mask.shape[0]):
            #axial cuts are made along the z axis with undersampling
            mask_2d = training_mask[k, ::ds, ::ds]
            image_2d = training_image[k, ::ds, ::ds]
            #we only recover the 2D sections containing the liver
            #if mask_2d contains only 0, it means that there is no liver
            #if len(np.unique(mask_2d)) != 1:
            masks_train.append(mask_2d)
            imgs_train.append(image_2d)

    imgs = np.ndarray(
            (len(imgs_train), image_rows, image_cols))
    imgs_mask = np.ndarray(
            (len(masks_train), image_rows, image_cols))

    for index, img in enumerate(imgs_train):
        imgs[index, :, :] = img

    for index, img in enumerate(masks_train):
        imgs_mask[index, :, :] = img
    print(imgs.shape)
    print(imgs_mask.shape)
    np.save('E:/CT_scans/Out/imgs_train.npy', imgs)
    np.save('E:/CT_scans/Out/masks_train.npy', imgs_mask)
    print('Saving to .npy files done.')

def load_train_data():
    imgs_train = np.load('E:/CT_scans/Out/imgs_train.npy')
    masks_train = np.load('E:/CT_scans/Out/masks_train.npy')
    return imgs_train, masks_train

def create_test_data():
    print('-'*30)
    print('Creating test data...')
    print('-'*30)
    images_test = os.listdir('E:/CT_scans/Out/Test')
    msks_test = os.listdir('E:/CT_scans/Out/Test_mask')
    imgs_test = []
    masks_test = []

    for image_name in images_test:
        print(image_name)
        img = np.load(os.path.join(test_data_path, image_name))
        print(img.shape)

        for k in range(img.shape[0]):
            img_2d = img[k,::ds, ::ds]
            imgs_test.append(img_2d)

    for image_name in msks_test:
        print(image_name)
        print('Yo')
        img = np.load(os.path.join(test_mask_data_path, image_name))
        print(img.shape)

        for k in range(img.shape[0]):
            img_2d = img[k,::ds, ::ds]
            masks_test.append(img_2d)

    imgst = np.ndarray(
            (len(imgs_test), image_rows, image_cols))
    imgs_maskt = np.ndarray(
            (len(masks_test), image_rows, image_cols))

    for index, img in enumerate(imgs_test):
        imgst[index, :, :] = img

    for index, img in enumerate(masks_test):
        imgs_maskt[index, :, :] = img

    np.save('E:/CT_scans/Out/imgs_test.npy', imgst)
    np.save('E:/CT_scans/Out/masks_test.npy', imgs_maskt)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    masks_test = np.load('masks_test.npy')
    return imgs_test, masks_test


if __name__ == '__main__':
    create_train_data()
    create_test_data()

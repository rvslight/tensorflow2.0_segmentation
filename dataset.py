import pandas as pd
import os
import tensorflow as tf
import cv2
import utils.config
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import numpy as np
import tensorflow.keras as keras
import cv2
import os
from imgaug import augmenters as iaa

# BASE_DIR = os.path.dirname(__file__)
# IMAGES_DIR = os.path.join(BASE_DIR, 'dataset', 'images')


class BatchGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    ##jitter is transform true or false
    def __init__(self, config, split, batch_size, shuffle=True, jitter=False):
        'Initialization'
        self.config = config
        self.split = split
        self.batch_size = batch_size

        self.image_h = config.DATA.IMG_H
        self.image_w = config.DATA.IMG_W
        self.n_channels = 3 ## TODO changed to config

        fold_df = pd.read_csv(self.config.FOLD_DF, engine='python')
        self.dataset = fold_df.loc[fold_df['split'] == self.split].reset_index(drop=True)

        if config.DEBUG:
            self.fold_df = self.fold_df[:100]
        print(self.split, 'set:', len(self.dataset))
        self.shuffle = shuffle

        self.on_epoch_end()
        self.jitter = jitter
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                # iaa.Fliplr(0.5), # horizontally flip 50% of all images
                # iaa.Flipud(0.2), # vertically flip 20% of all images
                # sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                #sometimes(iaa.Affine(
                    # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    # rotate=(-5, 5), # rotate by -45 to +45 degrees
                    # shear=(-5, 5), # shear by -16 to +16 degrees
                    # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                #)),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 3),
                           [
                               # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                               # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                               # search either for all edges or for directed edges
                               # sometimes(iaa.OneOf([
                               #    iaa.EdgeDetect(alpha=(0, 0.7)),
                               #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                               # ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               # iaa.Invert(0.05, per_channel=True), # invert color channels
                               iaa.Add((-10, 10), per_channel=0.5),
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               # change brightness of images (50-150% of original value)
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                               # iaa.Grayscale(alpha=(0.0, 1.0)),
                               # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                               # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(float(len(self.dataset)) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'

        '''
        l_bound = index*self.config['BATCH_SIZE']
        r_bound = (index+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']
        '''

        x_batch = np.zeros((self.batch_size, self.image_h, self.image_w, self.n_channels))
        y_batch = np.zeros((self.batch_size, self.image_h, self.image_w, 1))  # desired network output

        #current_batch = self.dataset[l_bound:r_bound]  전체 데이터에서 슬라이딩 연산자로 잘라가면서 batch단위로 데이터를 가지고옴..
        current_batch = self.dataset[index * self.batch_size:(index + 1) * self.batch_size]

        instance_num = 0
        current_batch = current_batch.T.to_dict().values() # dataframe to dictionnary.
        ## 그 중에서 하나의 데이터 열을 가져오면서 가종함.
        for instance in current_batch:
            img, mask = self.prep_image_and_mask(instance, jitter=self.jitter)

            # center of the bounding box is divided with the image width/height and grid width/height
            # to get the coordinates relative to a single element of a grid

            x_batch[instance_num] = img

            y_batch[instance_num] = tf.expand_dims(mask,axis=-1)

            instance_num += 1

        return x_batch, y_batch


    def prep_image_and_mask(self, dataset, jitter):
        fname = dataset['fname']
        try:
            path = os.path.join(self.config.DATA_DIR, 'images', fname)
            image = cv2.imread(path, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print('image_path:', path)

        try:
            path = os.path.join(self.config.DATA_DIR, 'masks', fname)
            mask = cv2.imread(path, 0)
        except:
            print('mask_path:', path)


        if self.config.DATA.RESIZE:
            image = cv2.resize(image,(self.config.DATA.IMG_W,self.config.DATA.IMG_H),interpolation=cv2.INTER_CUBIC) #Width / Height
            mask = cv2.resize(mask,(self.config.DATA.IMG_W,self.config.DATA.IMG_H),interpolation=cv2.INTER_LINEAR)
            mask = cv2.threshold(mask,127,255,0)[1]

        if jitter:
            image = self.aug_pipe.augment_image(image)

        ## normalization.
        image = image/ 255.

        mask = mask / 255

        return image, mask

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.dataset)


    def size(self):
        return len(self.dataset)

####### Old backup.. #############

class Dataset():
    def __init__(self, config, split, transform=None, display=False):
        self.config = config
        self.split = split
        self.transform = transform

        fold_df = pd.read_csv(self.config.FOLD_DF, engine='python')
        self.fold_df = fold_df.loc[fold_df['split'] == self.split].reset_index(drop=True)

        if config.DEBUG:
            self.fold_df = self.fold_df[:100]
        print(self.split, 'set:', len(self.fold_df))
        self.display = display

    def get_length(self):
        return len(self.fold_df)

    def DataGenerator(self,file_path, batch_size, shuffle = False):
        """
        generate image and mask at the same time
        use the same seed for image_datagen and mask_datagen
        to ensure the transformation for image and mask is the same
        """
        aug_dict = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
        aug_dict = dict(horizontal_flip=True,
                        fill_mode='nearest')

        image_datagen = ImageDataGenerator(**aug_dict)
        mask_datagen = ImageDataGenerator(**aug_dict)

        image_generator = image_datagen.flow_from_dataframe(self.fold_df,
                                                      directory=file_path+'/images',
                                                      x_col='fname',
                                                      y_col=None,
                                                      color_mode="rgb",
                                                      batch_size=batch_size,
                                                      target_size=(256, 256),
                                                      class_mode=None,
                                                      validate_filenames=False,
                                                      shuffle =shuffle,
                                                      seed=1)
        mask_generator = mask_datagen.flow_from_dataframe(self.fold_df,
                                                     directory=file_path+'/masks',
                                                     x_col='fname',
                                                     y_col=None,  # Or whatever
                                                     color_mode="grayscale",
                                                     batch_size=batch_size,
                                                     target_size=(256, 256),
                                                     class_mode=None,
                                                     validate_filenames=False,
                                                     shuffle=shuffle,
                                                     seed=1)
        ## validate_filenames=False for speedup..


        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            img = img / 255.
            mask = mask / 255.
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            yield (img, mask)


    def get_dataloader(self):

        image_list = []
        target_list = []
        for idx in range(len(self.fold_df)):
            item = self.getitem(idx)
            image_list.append(item[0])
            target_list.append(item[1])

        dataloader = tf.data.Dataset.from_tensor_slices((image_list,target_list))
        return dataloader, len(self.fold_df)

    def getitem(self, idx):
        fname = self.fold_df['fname'][idx]
        image = cv2.imread(os.path.join(self.config.DATA_DIR, 'images',fname),cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.config.DATA_DIR, 'masks',fname),cv2.IMREAD_GRAYSCALE)

        if self.config.DATA.RESIZE:
           image = cv2.resize(image,(self.config.DATA.IMG_W,self.config.DATA.IMG_H),interpolation=cv2.INTER_CUBIC)
           mask = cv2.resize(mask,(self.config.DATA.IMG_W,self.config.DATA.IMG_H),interpolation=cv2.INTER_LINEAR)
           mask = cv2.threshold(mask,127.5,255,0)[1]

        if self.transform is not None:
            iamge, mask = self.transform(image,mask)
        # cv2.imshow("test",image)
        # cv2.imshow("mask", mask)
        # cv2.waitKey()
        ## normailize..function..
        # normalization to (-1,1)
        image = (image / 255) - 1
        # if self.display is False:
        #     pass #normliaize..

        mask = mask / 255
        # to tf.tensor

        ## tf.cast
        image = tf.cast(image,tf.float32)
        mask = tf.cast(mask,tf.float32)

        return image, mask


if __name__ == '__main__':
    yml = 'configs/fastscnn_mv3_sj_add_data_1024.yml'
    config = utils.config.load(yml)
    dataset = Dataset(config, 'train', None)
    # dataloader = dataset.getDataloader()
    dataloader = dataset.DataGenerator("membrane/train", batch_size=2)
    for step, inputs in enumerate(dataloader):
        # input_image, target = tf.split(inputs, num_or_size_splits=[3, 3], axis=3)
        print(inputs[0].shape, inputs[1].shape)
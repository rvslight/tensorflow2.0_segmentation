import pandas as pd
import os
import tensorflow as tf
import cv2
import utils.config

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
    def getDataloader(self):

        image_list = []
        target_list = []
        for idx in range(len(self.fold_df)):
            item = self.getitem(idx)
            image_list.append(item[0])
            target_list.append(item[1])

        dataloader = tf.data.Dataset.from_tensor_slices((image_list,target_list))
        return dataloader

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
    dataloader = dataset.getDataloader()
    for step, inputs in enumerate(dataloader):
        # input_image, target = tf.split(inputs, num_or_size_splits=[3, 3], axis=3)
        print(inputs[0].shape, inputs[1].shape)
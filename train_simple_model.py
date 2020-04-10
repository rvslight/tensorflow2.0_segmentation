import numpy as np
import time
import os
import tensorflow as tf
import utils.config
from dataset import Dataset
from model import MyModel, UNet2
import segmentation_models as sm
import cv2
import random
from matplotlib import pyplot as plt
### 데이터셋 정의
### getItem쪽에 사용할 함수 정의
### pathlist를 정의하고.... tf.Tensor형태로 만듬..
### te.Tensor형태를 tuple로 묶어서 dataloader만듬.
### 모델 정의
### 로스(criterion)와 metric정의
### 에폭 정의
### trainloader로 순회하면서, 입력과 정답(타겟) 가져오고, gradients로 묶어서. tape에서 loss를 구함
### loss와 model의 trainablevariable을 입력으로 gradients를 구하고
### optimizer에 apply_gradient로 위에서 구한 gradient를 model.trainablevariable에 적용함
### 이후, 각각 위에서 정의한 loss, acc에 대한 result()값을 출력함

### https://github.com/dragen1860/TensorFlow-2.x-Tutorials/tree/master/14-Pixel2Pixel
##참고..

tf.random.set_seed(22)
np.random.seed(22)
os.environ['T_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true*y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true+y_pred, axis=-1)

    return 1 - (numerator + 1) / (denominator + 1)

def seed_everything():
    seed = 2019
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    yml = 'configs/fastscnn_mv3_sj_add_data_1024.yml'
    config = utils.config.load(yml)
    seed_everything()
    # train_dataloader, suffle_size = dataset.get_dataloader()
    # train_dataloader = train_dataloader.shuffle(suffle_size).batch(2)
    train_dataset = Dataset(config, 'train', None)
    train_dataloader = train_dataset.DataGenerator(config.DATA_DIR, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True)
    dataset_size = 100 # train_dataset.get_length()
    # model = MyModel()
    # h w c
    # model = UNet2(input_dims = [256,256,3], num_classes= 1)
    sm.set_framework('tf.keras') ## segmentation_model 2.0 support feature..
    model = sm.Unet('resnet18', input_shape=(256, 256, 3),  encoder_weights=None)#'imagenet')
    model.summary()

    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit_generator(train_dataloader, epochs=50)
    # model.save_weights("model.h5")

    epochs = config.TRAIN.NUM_EPOCHS
    loss_object = tf.keras.losses.BinaryCrossentropy()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    learning_rate = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    ### train
    for epoch in range(epochs):
        start = time.time()

        for step, (input_image,mask) in enumerate(train_dataloader):
            with tf.GradientTape() as grad_tape:
                output = model(input_image)
                # mask = tf.expand_dims(mask,-1)
                loss = loss_object(mask, output)

                # cv2.imshow('mask',mask.numpy()[0]*255)
                # cv2.waitKey()
                # print(loss.numpy())
                ## get a loss
            gradients = grad_tape.gradient(loss, model.trainable_variables)
            ## in grap.. get a gradient... 백워드 하는 것과 같음?? loss.backward()...
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
            train_accuracy(mask, output)
            print(epoch, "loss: ",train_loss.result().numpy(), "acc: ",train_accuracy.result().numpy(),"step: ",step)

            ## end of epoch. break..
            if step > dataset_size / config.TRAIN.BATCH_SIZE: break
    print('save_model')
    model.save_weights("model.h5")



    # model.fit_generator(train_dataloader, epochs=50)
    # model.save_weights("model.h5")

    ### test
    alpha = 0.3
    model.load_weights("model.h5")
    if not os.path.exists("./results"): os.mkdir("./results")

    for idx, (img, mask) in enumerate(train_dataloader):
        pred_mask = model(img).numpy()[0]
        pred_mask[pred_mask > 0.5] = 1
        pred_mask[pred_mask <= 0.5] = 0
        # img = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
        # H, W, C = img.shape
        # for i in range(H):
        #     for j in range(W):
        #         if pred_mask[i][j][0] <= 0.5:
        #             img[i][j] = (1 - alpha) * img[i][j] * 255 + alpha * np.array([0, 0, 255])
        #         else:
        #             img[i][j] = img[i][j] * 255
        # image_accuracy = np.mean(mask == pred_mask)
        # image_path = "./results/pred_" + str(idx) + ".png"
        # print("=> accuracy: %.4f, saving %s" % (image_accuracy, image_path))
        cv2.imshow("t_mask", np.uint8(mask[0]) * 255)
        cv2.imshow("mask", np.uint8(pred_mask* 255) )
        cv2.waitKey()
        # cv2.imwrite(image_path, img)
        # cv2.imwrite("./results/origin_%d.png" % idx, oring_img * 255)
        # if idx == 29: break




    # model.build(input_shape=(1, 224, 224, 3))
    # loss_object = dice_loss()#tf.keras.losses.(from_logits=True)
    # loss_object = tf.keras.losses.BinaryCrossentropy()
    # learning_rate = 1e-4
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    #
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    #
    # epochs = 30
    #
    # for epoch in range(epochs):
    #     start = time.time()
    #
    #     for step, inputs in enumerate(train_dataloader):
    #         input_image, mask = inputs[0], inputs[1]#, tf.cast(1,tf.int32)#tf.split(inputs, num_or_size_splits=[3,3], axis=3)
    #         # print(input_image.shape, mask.shape)
    #         cv2.imshow('input_image2', input_image.numpy()[0])
    #         # cv2.waitKey()
    #         with tf.GradientTape() as grad_tape:
    #             output = model(input_image)
    #             # label = tf.one_hot(label,depth=10)
    #             mask = tf.expand_dims(mask,-1)
    #             loss = loss_object(mask, output)
    #
    #             cv2.imshow('mask',mask.numpy()[0]*255)
    #             # cv2.waitKey()
    #             # print(loss.numpy())
    #             ## get a loss
    #         gradients = grad_tape.gradient(loss, model.trainable_variables)
    #         ## in grap.. get a gradient... 백워드 하는 것과 같음?? loss.backward()...
    #         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #         ## 해당 gradient를 적용함..  . optimzier.step..
    #         train_loss(loss)
    #         # train_accuracy(label, output)
    #         print("train_loss: "+str(train_loss.result()))
    #
    #         # shape = output.shape
    #         # key = 0
    #         if step % 40 == 0:
    #             pred_mask = output.numpy()[0].round()
    #             # pred_mask = np.transpose(pred_mask,axes=[1,2,0])
    #             pred_mask *= 255
    #             cv2.imshow('pred_mask',pred_mask)
    #             cv2.waitKey(100)
    #         # print("shape",shape)
    #         # print("train_accuracy: ",train_accuracy.result())
    #         # print("numpy_loss: "+str(train_loss.numpy()))
    #         # print(loss.numpy().shape)
    #         #
    #         # print(output.numpy())
    #         # print(output.numpy().shape)
    #
    #         # train_accuracy = train_accuracy(label,output)
    #         # if step % 100 == 0:
    #         #     print(epoch,step,float(train_loss),float(train_accuracy))
    #     # if epoch % 1 == 0:
    #     #     for step, inputs in enumerate(test_dataset):
    #     #         input_image, target = tf.split(inputs, num_or_size_splits=[3, 3], axis=3)
    #     print('Time taked for epoch {} is {} sec\n'.format(epoch+1, time.time()-start))

    # for inputs in test_dataset:
    #     input_image, target = tf.split(inputs, num_or_size_splits=[3,3],axis=3)
    #     print(input_image.shape, target.shape)
        
if __name__ == '__main__':
    main()



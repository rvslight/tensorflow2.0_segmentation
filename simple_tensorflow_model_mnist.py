from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.models import Model
import warnings

## 흐름.##
# 데이터 준비
# 정규화(노말라이제이션) 표준화는 평균으로부터 얼마나 떨어져 있는지를 나타내는 것이고, 정규화는.. 0,1사이로 바꾸는 것.. 즉 상대적 크기에 대한 영향력을 줄이는 것임..
# 채널 자원을 추가..1채널이기에.. ...,tf.newaxis
# dataset형태를 만들고.from_tensor_slices
# 모델을 만듬
# 로스와 옵티마이저를 정의하고
# train_loss, train_acc / test_loss, test_acc를 정의함
# train_step과 test_step을 정의함..
# train_step은 gradient_tape을 정의.. 모델에 데이터 넣고, 로스를 구하고, 기울기 구하고, 옵티버이저 업데이트, 트레인 로스와 트레인 정확도 계산..
# test_step은.. 모델에 데이터 넣고, 로스 구하고, 테스트 로스와 테스트 정확도 구함.
# 에폭 정의하고, 데이터 셋에서 ,,train_step, test_step돌면서 마지막에 프린트 찍어봄..

warnings.filterwarnings("ignore")
# load mnist
mninst = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mninst.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

#채널 자원을 추가 # 기존은 N,H,W임.. 1채널이기에.. N,H,W,C형태로.
x_train = x_train[..., tf.newaxis]

x_test = x_test[...,tf.newaxis]

#tf data 이용 데이터셋.
train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

#keras 이용 모델을 만듬
class MyModel(Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.conv1 = Conv2D(32,3,activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128,activation='relu')
        self.d2 = Dense(10,activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = MyModel()

# optimizer. loss fuc.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# tf gradient를 이용 훈련

@tf.function
def train_step(images,labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels,predictions) # loss계산시에 sparsecategoricalcro..를 썼기 때문에 . label이 one hot형태가 아닌 그냥 class값으로 와야 함.. prediction은 noehot형태.
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels,predictions) # label. is [2,1], prediction 은 noe hot encoding type이여도 무관함

# tf model test
@tf.function
def test_step(images,labels):
    predictions = model(images)
    t_loss = loss_object(labels,predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCH= 5

for epoch in range(EPOCH):
    for images, labels in train_ds:
        # print(f'labels: {labels}') # mnist는 라벨이 그대로 들어옴..
        # train_step(images,labels)
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels,
                               predictions)  # loss계산시에 sparsecategoricalcro..를 썼기 때문에 . label이 one hot형태가 아닌 그냥 class값으로 와야 함.. prediction은 noehot형태.
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)  # label. is [2,1], prediction 은 noe hot encoding type이여도 무관함

    for test_images, test_labels in test_ds:
        test_step(test_images,test_labels)

    print(f'epoch {epoch+1}, train_loss {train_loss.result()} train_acc {train_accuracy.result()*100} test_loss {test_loss.result()} test_acc {test_accuracy.result()*100}')

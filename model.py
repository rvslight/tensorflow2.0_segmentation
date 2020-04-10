# import segmentation_models as sm
from tensorflow import keras
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout, concatenate, UpSampling2D
### 모델을 케라스를 이용해서 정의하고..
## model
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = keras.layers.Conv2D(32,3,activation='relu')
        self.flatten = keras.layers.Flatten()
        self.d1 = keras.layers.Dense(128,activation='relu')
        self.d2 = keras.layers.Dense(10, activation='softmax')

        # initializer = tf.random_normal_initializer(0.,0.02)

    def call(self, x, training):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


class UNet():
    def __init__(self, num_class, image_size):
        # super(UNet,self).__init__()
        inputs = Input(shape=[image_size,image_size,3])


        c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = BatchNormalization()(c1)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        c1 = BatchNormalization()(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = BatchNormalization()(c2)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        c2 = BatchNormalization()(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = BatchNormalization()(c3)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        c3 = BatchNormalization()(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = BatchNormalization()(c4)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        c4 = BatchNormalization()(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = BatchNormalization()(c5)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        c5 = BatchNormalization()(c5)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = BatchNormalization()(c6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
        c6 = BatchNormalization()(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = BatchNormalization()(c7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
        c7 = BatchNormalization()(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = BatchNormalization()(c8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
        c8 = BatchNormalization()(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = BatchNormalization()(c9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        c9 = BatchNormalization()(c9)
        c10 = Conv2D(num_class, 1, activation='sigmoid')(c9)
        model = Model(inputs=inputs, outputs=c10)
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
        # return outputs

        # self.model = sm.Unet('resnet34', input_shape=(None, None, 6), encoder_weights=None)

    def getModel(self):
        return self.model




class UNet2(Model):
    """A Unet model.
    Takes input from the user regarding key design choices (e.g. input dimensions,
    kernel size), and then builds a Unet to those specs. Since we're inheriting
    from `keras.models.Model`, the resulting object can be manipulated using the
    Keras Model API.
    """

    def __init__(self, input_dims, num_classes,
                 filters_list=[64, 128, 256, 512, 1024], kernel_size=(3, 3),
                 pool_size=(2, 2), crop=False, activation='relu'):
        super(UNet2,self).__init__()
        """Creates a Unet based on user input.
        Following the instantiation of a Unet object, the user can run
        `model.summary()` to confirm that the architecture is as expected. He can
        then compile and fit the model as he would with any other Keras model.
        Parameters
        ----------
        input_dims: tuple
           a tuple specifying the input dimension of the training set:
           (height, weight, num_channels)
        num_classes: int
           number of output classes
        filters_list: list of ints
           a list of integers specifying the number of convolutional filters in
           each block. Note that the length of this list determines depth.
        kernel_size: tuple
           kernel_size as in the Keras API
        pool_size: tuple
           pool_size as in the Keras API
        crop: bool
           boolean denoting whether we should crop the output of the encoder
           (as in the original paper). Note that this value also determines
           whether we perform "same" or "valid" convolution. If `crop=True` we
           perform valid convolution, otherwise we do same convolution.
        activation: str
           activation as in the Keras API
        """

        self.input_dims = input_dims
        self.num_classes = num_classes
        self.filters_list = filters_list
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.activation = activation
        self.crop = crop
        self.padding = "valid" if crop else "same"
        self.depth = len(filters_list)
        # self.cropping_dims_list = self._calc_cropping_dims()
        self.encoder_block_outputs = []
        self.idx = 0

        self._build_unet()

    def _build_unet(self):
        """Builds the Unet model.
         """
        inputs = Input(self.input_dims)
        enc = self._build_encoder(inputs)
        unet = self._build_decoder(enc)

        super().__init__(inputs=inputs, outputs=unet)

    def _build_encoder(self, input):
        """Builds the encoder.
        """
        enc = input
        for i in range(self.depth - 1):
            enc = self._encoder_block(enc)
        enc = self._double_conv(enc, self.filters_list[-1])

        return enc

    def _build_decoder(self, input):
        """Builds the decoder.
        """
        dec = input
        for i in range(self.depth - 1):
            dec = self._decoder_block(dec)
        dec = Conv2D(self.num_classes, (1, 1), activation="sigmoid")(dec)

        return dec

    def _encoder_block(self, input):
        """Encoder block.
        """
        conv = self._double_conv(input, self.filters_list[self.idx])
        enc = MaxPooling2D(pool_size=self.pool_size, padding="same")(conv)
        # if self.crop:
        #     cropping_dims = self.cropping_dims_list[self.idx]
        #     conv = Cropping2D(cropping=cropping_dims)(conv)

        self.encoder_block_outputs.append(conv)
        self.idx += 1

        return enc

    def _decoder_block(self, input):
        """Decoder block.
        """
        self.idx -= 1

        filters = self.filters_list[self.idx]

        dec = Conv2DTranspose(filters, kernel_size=self.pool_size,
                              strides=self.pool_size, padding="valid",
                              activation=self.activation)(input)
        try:
            dec = Concatenate()([dec, self.encoder_block_outputs[self.idx]])
        except ValueError:
            raise ValueError("Input dimensions and architecture are not compatible. Try input_dims = {}".format(
                self._suggest_input_dims()))
        dec = self._double_conv(dec, filters)

        return dec

    def _double_conv(self, input, filters):
        """Back-to-back convolutions
        """
        conv = input
        for i in range(2):
            conv = Conv2D(filters, self.kernel_size, padding=self.padding,
                          activation=self.activation)(conv)

        return conv
import tensorflow as tf
import segmentation_models as sm
saved_model_dir = 'h5_model'
sm.set_framework('tf.keras')  ## segmentation_model 2.0 support feature..

backbone = 'mobilenetv2'
model = sm.Unet(backbone, input_shape=(256, 256, 3), encoder_weights=None, activation='sigmoid')#activation='identity')#, decoder_attention_type='scse')  # 'imagenet')
model.summary()

file_name = "0_0.9767963647842407_model.h5"
model.load_weights(file_name)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("./tflite/"+file_name.split(".")[0]+".tflite", "wb").write(tflite_model)
print('done')
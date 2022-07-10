#%% Imports


import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
from tensorflow.keras.layers import *

# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# #%%Set mixed precision for NVidia Tensor cores
# from tensorflow.keras import mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

#tf.get_logger().setLevel('WARNING')

#%% Load datasets
PATH = './GramianAngularFields'
SYMBOL = 'SPY'


validation_dir = os.path.join(PATH,f'{SYMBOL}-STK-SMART-USD', 'TRAIN')
test_dir = os.path.join(PATH,f'{SYMBOL}-STK-SMART-USD', 'TEST')
fine_tune = False

#HUB_LINK = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2"
#HUB_LINK = "https://tfhub.dev/google/imagenet/resnet_v2_152/classification/5"
#https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/5
#"https://tfhub.dev/google/imagenet/resnet_v2_152/feature-vector/4
BATCH_SIZE = 300
model_name = "bit_s-r50x1" # @param ['efficientnetv2-s', 'efficientnetv2-m', 'efficientnetv2-l', 'efficientnetv2-s-21k', 'efficientnetv2-m-21k', 'efficientnetv2-l-21k', 'efficientnetv2-xl-21k', 'efficientnetv2-b0-21k', 'efficientnetv2-b1-21k', 'efficientnetv2-b2-21k', 'efficientnetv2-b3-21k', 'efficientnetv2-s-21k-ft1k', 'efficientnetv2-m-21k-ft1k', 'efficientnetv2-l-21k-ft1k', 'efficientnetv2-xl-21k-ft1k', 'efficientnetv2-b0-21k-ft1k', 'efficientnetv2-b1-21k-ft1k', 'efficientnetv2-b2-21k-ft1k', 'efficientnetv2-b3-21k-ft1k', 'efficientnetv2-b0', 'efficientnetv2-b1', 'efficientnetv2-b2', 'efficientnetv2-b3', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'bit_s-r50x1', 'inception_v3', 'inception_resnet_v2', 'resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152', 'resnet_v2_50', 'resnet_v2_101', 'resnet_v2_152', 'nasnet_large', 'nasnet_mobile', 'pnasnet_large', 'mobilenet_v2_100_224', 'mobilenet_v2_130_224', 'mobilenet_v2_140_224', 'mobilenet_v3_small_100_224', 'mobilenet_v3_small_075_224', 'mobilenet_v3_large_100_224', 'mobilenet_v3_large_075_224']

model_handle_map = {
  "efficientnetv2-s": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2",
  "efficientnetv2-m": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/feature_vector/2",#0.56
  "efficientnetv2-l": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/feature_vector/2",
  "efficientnetv2-s-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/feature_vector/2",
  "efficientnetv2-m-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_m/feature_vector/2",
  "efficientnetv2-l-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/feature_vector/2",
  "efficientnetv2-xl-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2",
  "efficientnetv2-b0-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2",
  "efficientnetv2-b1-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b1/feature_vector/2",
  "efficientnetv2-b2-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b2/feature_vector/2",
  "efficientnetv2-b3-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b3/feature_vector/2",
  "efficientnetv2-s-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_s/feature_vector/2",
  "efficientnetv2-m-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_m/feature_vector/2",
  "efficientnetv2-l-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_l/feature_vector/2",
  "efficientnetv2-xl-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2",
  "efficientnetv2-b0-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/feature_vector/2",
  "efficientnetv2-b1-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/feature_vector/2",
  "efficientnetv2-b2-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/feature_vector/2",
  "efficientnetv2-b3-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/feature_vector/2",
  "efficientnetv2-b0": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2",
  "efficientnetv2-b1": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b1/feature_vector/2",
  "efficientnetv2-b2": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b2/feature_vector/2",
  "efficientnetv2-b3": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b3/feature_vector/2",
  "efficientnet_b0": "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
  "efficientnet_b1": "https://tfhub.dev/tensorflow/efficientnet/b1/feature-vector/1",
  "efficientnet_b2": "https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1",
  "efficientnet_b3": "https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1",
  "efficientnet_b4": "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1",
  "efficientnet_b5": "https://tfhub.dev/tensorflow/efficientnet/b5/feature-vector/1",
  "efficientnet_b6": "https://tfhub.dev/tensorflow/efficientnet/b6/feature-vector/1",
  "efficientnet_b7": "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1",
  "bit_s-r50x1": "https://tfhub.dev/google/bit/s-r50x1/1",
  "inception_v3": "https://tfhub.dev/google/imagenet/inception_v3/feature-vector/4",
  "inception_resnet_v5": "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5",
  "resnet_v1_50": "https://tfhub.dev/google/imagenet/resnet_v1_50/feature-vector/5",
  "resnet_v1_101": "https://tfhub.dev/google/imagenet/resnet_v1_101/feature-vector/5",
  "resnet_v1_152": "https://tfhub.dev/google/imagenet/resnet_v1_152/feature-vector/5",
  "resnet_v2_50": "https://tfhub.dev/google/imagenet/resnet_v2_50/feature-vector/5",
  "resnet_v2_101": "https://tfhub.dev/google/imagenet/resnet_v2_101/feature-vector/5",
  "resnet_v2_152": "https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/5",#https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/5
  "nasnet_large": "https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/4",
  "nasnet_mobile": "https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4",
  "pnasnet_large": "https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/4",
  "mobilenet_v2_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
  "mobilenet_v2_130_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4",
  "mobilenet_v2_140_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4",
  "mobilenet_v3_small_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5",
  "mobilenet_v3_small_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/feature_vector/5",
  "mobilenet_v3_large_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5",
  "mobilenet_v3_large_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5",
}

model_image_size_map = {
  "efficientnetv2-s": 384,
  "efficientnetv2-m": 480,
  "efficientnetv2-l": 480,
  "efficientnetv2-b0": 224,
  "efficientnetv2-b1": 240,
  "efficientnetv2-b2": 260,
  "efficientnetv2-b3": 300,
  "efficientnetv2-s-21k": 384,
  "efficientnetv2-m-21k": 480,
  "efficientnetv2-l-21k": 480,
  "efficientnetv2-xl-21k": 512,
  "efficientnetv2-b0-21k": 224,
  "efficientnetv2-b1-21k": 240,
  "efficientnetv2-b2-21k": 260,
  "efficientnetv2-b3-21k": 300,
  "efficientnetv2-s-21k-ft1k": 384,
  "efficientnetv2-m-21k-ft1k": 480,
  "efficientnetv2-l-21k-ft1k": 480,
  "efficientnetv2-xl-21k-ft1k": 512,
  "efficientnetv2-b0-21k-ft1k": 224,
  "efficientnetv2-b1-21k-ft1k": 240,
  "efficientnetv2-b2-21k-ft1k": 260,
  "efficientnetv2-b3-21k-ft1k": 300, 
  "efficientnet_b0": 224,
  "efficientnet_b1": 240,
  "efficientnet_b2": 260,
  "efficientnet_b3": 300,
  "efficientnet_b4": 380,
  "efficientnet_b5": 456,
  "efficientnet_b6": 528,
  "efficientnet_b7": 600,
  "inception_v3": 299,
  "inception_resnet_v5": 299,
  "nasnet_large": 331,
  "pnasnet_large": 331,
}

model_handle = model_handle_map.get(model_name)
pixels = model_image_size_map.get(model_name, 224)

print(f"Selected model: {model_name} : {model_handle}")

IMAGE_SIZE = (pixels, pixels)
print(f"Input size {IMAGE_SIZE}")


data_dir = os.path.join(PATH,f'{SYMBOL}-STK-SMART-USD', 'TRAIN')

def build_dataset(subset):
  return tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=.50,
      subset=subset,
      label_mode="categorical",
      class_names=['LONG', 'SHORT','FLAT'],
      # Seed needs to provided when using validation_split and shuffle = True.
      # A fixed seed is used so that the validation set is stable across runs.
      shuffle=True,
      seed=123,
      image_size=IMAGE_SIZE,
      batch_size=1)

train_ds = build_dataset("training")
class_names = tuple(train_ds.class_names)
train_size = train_ds.cardinality().numpy()
train_ds = train_ds.unbatch().batch(BATCH_SIZE)
train_ds = train_ds.repeat()

normalization_layer = tf.keras.layers.Rescaling(1. / 255)
preprocessing_model = tf.keras.Sequential([normalization_layer])

train_ds = train_ds.map(lambda images, labels:
                        (preprocessing_model(images), labels))

val_ds = build_dataset("validation")
valid_size = val_ds.cardinality().numpy()
val_ds = val_ds.unbatch().batch(BATCH_SIZE)
val_ds = val_ds.map(lambda images, labels:
                    (normalization_layer(images), labels))

# validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
#                                                                  shuffle=True,c
#                                                                  batch_size=BATCH_SIZE,
#                                                                  image_size=IMAGE_SIZE)

test_ds = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                      shuffle=False,
                                                      batch_size=BATCH_SIZE,
                                                      image_size=IMAGE_SIZE,
                                                      class_names=['LONG', 'SHORT', 'FLAT'],
                                                      label_mode="categorical")

test_size = test_ds.cardinality().numpy()
test_ds = test_ds.unbatch().batch(BATCH_SIZE)
test_ds = test_ds.map(lambda images, labels:
                    (normalization_layer(images), labels))

# %%Use buffered prefetching to load images from disk without having I/O become blocking. 
# To learn more about this method see the data performance guide:
# (https://www.tensorflow.org/guide/data_performance)
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

#%% Use data augmentation
# data_augmentation = tf.keras.Sequential([
#   tf.keras.layers.RandomFlip('horizontal'),
#   tf.keras.layers.RandomRotation(0.2),
# ])
# %% Create Model
def create_thub_model():
  print("Building model with", model_name)
  print("Model handle", model_handle)
  base_model = tf.keras.Sequential([
      tf.keras.Input(shape=IMAGE_SIZE + (3,)),
      tf.keras.layers.Rescaling(1./255, offset=0),
      hub.KerasLayer(model_handle, trainable=fine_tune),
      tf.keras.layers.Dense(100,activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10,activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(3)
  ])
  base_learning_rate = 0.05
  opt1 = tf.keras.optimizers.SGD(learning_rate=base_learning_rate, momentum=0.9)
  opt2 = tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
  base_model.compile(optimizer=opt1,
              #loss=tf.keras.losses.BinaryCrossentrophy(from_logits=True, label_smoothing=0.1),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              #loss='binary_crossentropy',
              #metrics='acc')
              metrics=[tf.keras.metrics.CategoricalAccuracy()])
  
  base_model.summary()
  return base_model

def create_cnn_model():
  base_model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255, offset=0),
    #  First Convolution
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(512, 512, 3)),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    # Second Convolution
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    # Third Convolution
    Conv2D(256, kernel_size=4, activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dropout(0.4),
    # Output layer
    Dense(1, activation='sigmoid')]
  )
    
  base_learning_rate = 0.01
  opt1 = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9)
  opt = tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
  #opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

  # base_model.compile(opt1,
  #             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  #             metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]) #
  
  base_model.compile(optimizer=opt,
              #loss=tf.keras.losses.BinaryCrossentrophy(from_logits=True, label_smoothing=0.1),
              loss='binary_crossentropy',
              metrics='acc')
              #metrics=[tf.keras.metrics.CategoricalAccuracy()])
  return base_model

#%% Set checkpoint
checkpoint_filepath = './Model/weights/best_weights'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_categorical_accuracy',
    mode='max',
    save_best_only=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=5, min_lr=0.00000001)

early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
# %% Train the model
#After training for 10 epochs, you should see ~94% accuracy on the validation set.
initial_epochs = 300

#model = create_cnn_model()
model = create_thub_model()
loss0, accuracy0 = model.evaluate(val_ds)

steps_per_epoch = max(1, train_size // BATCH_SIZE)
validation_steps = max(1, valid_size // BATCH_SIZE)

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# hist = model.fit(
#     train_ds,
#     epochs=5, steps_per_epoch=steps_per_epoch,
#     validation_data=val_ds,
#     validation_steps=validation_steps).history

history = model.fit(train_ds,
                    epochs=initial_epochs,
                    validation_data=val_ds,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=[model_checkpoint_callback, reduce_lr, early_stop_callback])

#%% Load best weights
model.load_weights(checkpoint_filepath)



loss1, accuracy1 = model.evaluate(val_ds)
#print(f'M0 Loss ({loss0}), Accuracy ({accuracy0})')
print(f'M1 {model_name} Loss ({loss1}), Accuracy ({accuracy1})')

model.save(f"./Models/models/{model_name}-{SYMBOL}_model_{datetime.now().strftime('%Y-%m-%d')}.{'TUNED' if fine_tune else 'STOCK'}")

#%%
saved_model = tf.keras.models.load_model(f"./Models/models/{model_name}-{SYMBOL}_model_{datetime.now().strftime('%Y-%m-%d')}.{'TUNED' if fine_tune else 'STOCK'}")
lossL, accuracyL = saved_model.evaluate(val_ds)

lossT, accuracyT = saved_model.evaluate(test_ds)

#print(f'M0 Loss ({loss0}), Accuracy ({accuracy0})')
#print(f'M1 {model_name} Loss ({loss1}), Accuracy ({accuracy1})')
print(f'L1 {model_name} Loss ({lossL}), Accuracy ({accuracyL})')
print(f'T1 {model_name} Loss ({lossT}), Accuracy ({accuracyT})')

print ('DONE!!!')
# %%Learning curves
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.ylabel('Accuracy')
# plt.ylim([min(plt.ylim()),1])
# plt.title('Training and Validation Accuracy')

# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross Entropy')
# plt.ylim([0,1.0])
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.show()
# %%

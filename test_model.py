#%% Imports
import os
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime



#%% Load data
PATH = './GramianAngularFields'
SYMBOL = 'SPY'
MODEL_NAME="bit_s-r50x1"
TUNED=False

train_dir = os.path.join(PATH, f'{SYMBOL}-STK-SMART-USD','TRAIN')
test_dir = os.path.join(PATH,f'{SYMBOL}-STK-SMART-USD', 'TEST')

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
#%% Load images
def build_dataset(subset, val_split):
  return tf.keras.preprocessing.image_dataset_from_directory(
      train_dir,
      validation_split=val_split,
      subset=subset,
      label_mode="categorical",
      class_names=['LONG', 'SHORT','FLAT'],
      # Seed needs to provided when using validation_split and shuffle = True.
      # A fixed seed is used so that the validation set is stable across runs.
      shuffle=False,
      seed=123,
      image_size=IMAGE_SIZE,
      batch_size=1)

train_ds = build_dataset("training", 0.8)
class_names = tuple(train_ds.class_names)
train_size = train_ds.cardinality().numpy()
train_ds = train_ds.unbatch().batch(BATCH_SIZE)
# train_ds = train_ds.repeat()

normalization_layer = tf.keras.layers.Rescaling(1. / 255)
preprocessing_model = tf.keras.Sequential([normalization_layer])

train_ds = train_ds.map(lambda images, labels:
                        (preprocessing_model(images), labels))

val_ds = build_dataset("validation", 0.8)
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
                                                      label_mode="categorical",
                                                      class_names=['LONG', 'SHORT', 'FLAT'])

test_size = test_ds.cardinality().numpy()
test_ds = test_ds.unbatch().batch(BATCH_SIZE)
test_ds = test_ds.map(lambda images, labels:
                    (normalization_layer(images), labels))


print('Number of validation batches: %d' % tf.data.experimental.cardinality(val_ds))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_ds))
# %%Use buffered prefetching to load images from disk without having I/O become blocking. 
# To learn more about this method see the data performance guide:
# (https://www.tensorflow.org/guide/data_performance)
# AUTOTUNE = tf.data.AUTOTUNE

# train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
# validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
# test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
# rescale = tf.keras.layers.Rescaling(1./127.5, offset=0)


# %% This feature extractor converts each 160x160x3 image into a 5x5x1280 block of features. 
# Let's see what it does to an example batch of images:

#mirrored_strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

#mirrored_strategy = tf.distribute.MirroredStrategy()




# #%% Train more...

# initial_epochs = 3
# steps_per_epoch = train_size // BATCH_SIZE
# validation_steps = valid_size // BATCH_SIZE

# saved_model.fit(train_ds,
#                     epochs=initial_epochs,
#                     steps_per_epoch=steps_per_epoch,
#                     validation_data=val_ds,
#                     batch_size=100000)


# %%
# saved_model.save(f"./Models/models/tV2_model_{datetime.now().strftime('%Y-%m-%d')}")
saved_model = tf.keras.models.load_model(f"./Models/models/{MODEL_NAME}-{SYMBOL}_model_2022-05-25.{'TUNED' if TUNED else 'STOCK'}")
# %%
lossL, accuracyL = saved_model.evaluate(val_ds)
print('Restored model, accuracy: {:5.2f}%'.format(100 * accuracyL))
#%% Test dataset
lossT, accuracyT = saved_model.evaluate(test_ds)
print('Restored model, accuracy: {:5.2f}%'.format(100 * accuracyT))

# %%
pred_float = saved_model.predict(test_ds)
# test_label = np.concatenate([y for x, y in test_data], axis=0) 
# %%
import numpy as np

predictions = np.array([])
labels =  np.array([])
for x, y in train_ds:
  predictions = np.concatenate([predictions, np.argmax(saved_model.predict(x), axis = -1)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
  print(f"X: {x}, Y:{y}")
# %%
trading_calendar = []
for currentpath, folders, files in os.walk(os.path.join(PATH,f'{SYMBOL}-STK-SMART-USD', 'TRAIN')):
  trading_calendar.extend(
      datetime.strptime(
          file.replace(".png", "").replace("_", "-"), "%Y-%m-%d")
      for file in files)

trading_calendar.sort()
# %%
import pandas as pd
tc_df = pd.DataFrame(zip(trading_calendar, predictions), columns=['DateTime', 'Prediction'])
tc_df.DateTime = pd.to_datetime(tc_df.DateTime, utc=True)
tc_df = tc_df.set_index('DateTime')
#Replace 0 with -1
tc_df.Prediction = np.where(tc_df.Prediction == 0, -1, 1)
# %%
import images_from_data as ifd
HOST="http://192.168.2.40:8086"
ORG="MarketData"
BUCKET="market_data"
TOKEN ="-hzZ7-09Y_NWKbjPFIVRVxeyOoK0P857Ype1guP3sk5lisZDT-erGJZ9F2oxaXD554H31glI0dJSQRbESICdvg=="

START_DATE = datetime(2010,1,1)
END_DATE = datetime(2023,1,1)

data_df = ifd.load_data_form_idb(symbol=f'{SYMBOL}-STK-SMART-USD', measurement='1 day', start_date=START_DATE, end_date=END_DATE)
data_df = data_df.set_index('DateTime')


# %%
results_df = pd.merge(
  left=data_df, 
  right=tc_df, 
  how='left', 
  left_index=True, 
  right_index=True).fillna(method='ffill')

results_df['Return'] = results_df['close'].pct_change()*results_df['Prediction']
results_df['TotalReturn'] = results_df.Return.cumsum()  
# %%
import matplotlib.pyplot as mp
results_df.plot(y='TotalReturn')
mp.savefig(f'{SYMBOL}_TotalReturn.png')
# %%

from keras.models import Sequential
from keras.layers import Lambda, Dense, Dropout, Flatten, ELU, MaxPooling2D
from keras.layers.convolutional import Convolution2D, Convolution3D
from keras.callbacks import  ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from ImageProcessor import load_driving_log
from ImageProcessor import ImageGenerator
import json

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(66,200,3),output_shape=(66,200,3)))
model.add(Convolution2D(24,5,5,border_mode='valid', activation='linear', subsample=(2,2),input_shape=(66,200,3)))
model.add(ELU())
model.add(Convolution2D(36,5,5,border_mode='valid', activation='linear', subsample=(2,2)))
model.add(ELU())
model.add(Convolution2D(48,5,5,border_mode='valid', activation='linear', subsample=(2,2)))
model.add(ELU())
model.add(Convolution2D(64,3,3,border_mode='valid', activation='linear', subsample=(1,1)))
model.add(ELU())
model.add(Convolution2D(64,3,3,border_mode='valid', activation='linear', subsample=(1,1)))
model.add(ELU())
model.add(Flatten())
model.add(Dense(100, activation='linear'))
model.add(ELU())
model.add(Dense(50, activation='linear'))
model.add(ELU())
model.add(Dense(10, activation='linear'))
model.add(ELU())
model.add(Dense(1))

model.compile(optimizer=Adam(lr=0.0001), loss="mse")
print(model.count_params())
print(model.summary())
with open('model.json', "w") as outfile:
    json.dump(model.to_json(), outfile)


#dirs = "udacity_data"
dirs = "own_data"

log_data = []
for d in dirs.split(','):
    log_data += load_driving_log(d)

data_train, data_val = train_test_split(log_data, test_size=2048, random_state=42)
val_gen = ImageGenerator(data_val, augment_data=False)
train_gen = ImageGenerator(data_train, augment_data=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1,min_lr=1e-7)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
model_checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

model.fit_generator(\
  train_gen,\
  #samples_per_epoch=22400,\
  samples_per_epoch=82304,\
  nb_epoch=10,\
  validation_data=val_gen,
  nb_val_samples=len(data_val),
  max_q_size=1024,
  nb_worker=8,
  pickle_safe=True,
  callbacks=[reduce_lr, model_checkpoint])

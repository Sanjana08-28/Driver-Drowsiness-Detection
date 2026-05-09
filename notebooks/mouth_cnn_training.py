import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout

from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt


# SETTINGS
IMG_SIZE = 224

BATCH_SIZE = 32


# DATASET PATHS
train_dir = '../datasets/eye_dataset/dataset_new/mouth_dataset/train'

validation_dir = '../datasets/eye_dataset/dataset_new/mouth_dataset/validation'

test_dir = '../datasets/eye_dataset/dataset_new/mouth_dataset/test'


# IMAGE AUGMENTATION
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)


# LOAD DATA
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)


# LOAD MOBILENETV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False


# BUILD MODEL
model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dropout(0.5),

    Dense(1, activation='sigmoid')

])


# COMPILE MODEL
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# TRAIN MODEL
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5
)


# SAVE MODEL
model.save('../models/mobilenetv2_mouth_model.h5')

print("MOUTH MODEL SAVED SUCCESSFULLY")


# PLOT ACCURACY
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Mouth CNN Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['Train', 'Validation'])

plt.show()
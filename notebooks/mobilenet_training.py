import tensorflow as tf


from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout

from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# IMAGE SETTINGS
IMG_SIZE = 224

BATCH_SIZE = 32


# DATASET PATHS
train_dir = '../datasets/eye_dataset/dataset_new/train'

validation_dir = '../datasets/eye_dataset/dataset_new/validation'

test_dir = '../datasets/eye_dataset/dataset_new/test'


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


# LOAD DATASETS
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
model.save('../models/mobilenetv2_eye_model.h5')

print("MODEL SAVED SUCCESSFULLY")


# PLOT ACCURACY
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['Train', 'Validation'])

plt.savefig("../outputs/graphs/eye_accuracy.png")

plt.show()

# ==========================================
# CONFUSION MATRIX + CLASSIFICATION REPORT
# ==========================================

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

# RESET TEST GENERATOR
test_generator.reset()

# PREDICTIONS
predictions = model.predict(test_generator)

y_pred = np.argmax(predictions, axis=1)

y_true = test_generator.classes

# CLASS NAMES
class_names = list(test_generator.class_indices.keys())

# PRINT REPORT
print("\nCLASSIFICATION REPORT\n")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print(report)

with open("../outputs/reports/eye_classification_report.txt", "w") as f:
    f.write(report)

# CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred)

# PLOT
plt.figure(figsize=(6,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Eye Model Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig("../outputs/graphs/eye_confusion_matrix.png")

plt.show()

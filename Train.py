from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from numpy import random
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
tf.random.set_seed(10)
random.seed(10)

# GLOBALS--------------------------------------------------------
LR = 1e-4
EPOCHS = 19
BS = 32
# LOADING THE DATASET---------------------------------------------
print("loading images...")
folder = 'C:/Users/moham/OneDrive/Desktop/5/PMDL/Project/dataset'

data = []
labels = []

for foldername in os.listdir(folder):
    for filename in os.listdir(os.path.join(folder, foldername)):
        image = load_img(os.path.join(folder, foldername, filename), target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)
        label = foldername

        data.append(image)
        labels.append(label)
# IMAGES PREPROCESSING-----------------------------------------------
data = np.array(data, dtype="float32")
labels = np.array(labels)

# CATEGORIZE THE LABELS -----------------------------------------------
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# 80% TRAINING & 20% VALIDATING-------------------------------------------
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=10)

print("Train size: ", X_train.shape)
print("Train Label size: ", Y_train.shape)
print("Validation size: ", X_test.shape)
print("Validation Label size: ", Y_test.shape)

# DATA AUGMENTATION-----------------------------------------------------
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# LOAD THE BASE MODEL----------------------------------------------------
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

for layer in baseModel.layers:
	layer.trainable = False

# CREATE THE HEAD MODEL TO BE TRAINED--------------------------------------
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256)(headModel)
headModel = LeakyReLU(alpha=0.1)(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(256)(headModel)
headModel = LeakyReLU(alpha=0.1)(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)
model.summary()

# COMPILING & TRAINING MODEL-------------------------------------------------
print("compiling model...")
opt = Adam(lr=LR, decay=LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("training head model...")
H = model.fit(
	aug.flow(X_train, Y_train, batch_size=BS),
	validation_data=(X_test, Y_test),
	epochs=EPOCHS)

# EVALUATE MODEL & PRINT REPORT-----------------------------------------------
print("evaluating network...")
pred = model.predict(X_test, batch_size=BS)
predIdxs = np.argmax(pred, axis=1)

print(classification_report(Y_test.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# SAVE MODEL-------------------------------------------------------------------
print("saving mask detector model...")
model.save("mask_detector_model", save_format="h5")

# PLOTTING----------------------------------------------------------------------
fig1, ax1 = plt.subplots()
ax1.plot(range(EPOCHS), H.history["loss"], label = "Training Loss")
ax1.plot(range(EPOCHS), H.history["val_loss"], label = "Validation Loss")
ax1.legend(loc = "upper left")
ax1.grid()
ax1.set_xlabel('Epoch')
ax1.set_title('Cost')

fig2, ax2 = plt.subplots()
ax2.plot(range(EPOCHS), H.history["accuracy"], label = "Training Accuracy")
ax2.plot(range(EPOCHS), H.history["val_accuracy"], label = "Validation Accuracy")
ax2.legend(loc = "upper left")
ax2.grid()
ax2.set_xlabel('Epoch')
ax2.set_title('Accuracy')

plt.show()
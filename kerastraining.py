import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

# Define parameters
input_shape = (150, 150, 3)  # Input shape of images
batch_size = 32  # Number of samples per gradient update
epochs = 30  # Number of epochs to train the model

# Define data generators for training and validation data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('data/train', target_size=input_shape[:2], batch_size=batch_size, class_mode='categorical')
test_generator = test_datagen.flow_from_directory('data/test', target_size=input_shape[:2], batch_size=batch_size, class_mode='categorical')

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(train_generator.num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
checkpoint = ModelCheckpoint('keras_Model.h5', monitor='val_loss', save_best_only=True)
history = model.fit(train_generator, epochs=epochs, validation_data=test_generator, callbacks=[checkpoint])

# Save the label names
label_names = sorted(os.listdir('data/train'))
with open('labels.txt', 'w') as f:
    f.write('\n'.join(label_names))

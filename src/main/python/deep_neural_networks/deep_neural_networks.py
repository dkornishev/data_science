
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def metrics_score(actual, predicted):
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='.0f', xticklabels=class_names_list, yticklabels=class_names_list)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


model1 = Sequential()

# Generating the model and adding layers
# Two convolution layers and pooling layer
model1.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last', input_shape=(28,28,1)))
model1.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the model and making room for the fully connected component
model1.add(Flatten())
# TODO Use leaky relu
model1.add(Dense(64, activation="relu"))
model1.add(Dropout(0.1))
model1.add(Dense(10, activation="softmax"))

model1.summary()

optimizer = Adam(learning_rate=0.01)

model1.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

model1.fit(x_train, y_train, validation_data=(x_test, y_test), verbose = 1, batch_size=128, epochs = 10)

test_pred1 = np.argmax(model1.predict(x_test), axis=-1)

metrics_score(testY, test_pred1)
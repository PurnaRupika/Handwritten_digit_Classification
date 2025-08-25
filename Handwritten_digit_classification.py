import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

(X_train,y_train) , (X_test,y_test) = keras.datasets.mnist.load_data()

len(X_train)

len(X_test)

X_train[0].shape   # 28 * 28 pixel image

plt.matshow(X_train[0])

# scaling the array values to [0,1]
X_train = X_train / 255
X_test = X_test / 255

X_train[0]  # number 5 represented in a 2-D array where 0 = black, 255 = white

y_train[0]

X_train.shape

# flattening the 28 * 28 pixels image into 1-D array with dimension 784
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)

X_train_flattened.shape

X_test_flattened.shape

X_train_flattened[0]

# initially using only input layer with 784 neurons without any hidden layers
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

# sparse_categorical_crossentropy is used when the output variable(y_train) has integer values (in this dataset 0 to 9 digits)
# categorical_crossentropy is used when the output variable(y_train) is represented in one-hot encoder
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train_flattened, y_train, epochs=5)

model.evaluate(X_test_flattened, y_test)

plt.matshow(X_test[0])

y_predict = model.predict(X_test_flattened)
y_predict[0]  # this gives 10 values representing the scores of every digit from 0 to 9 being outputs

np.argmax(y_predict[0])
# for printing the score that is maximum out of all as it represents the chance of being correct...here it is 9.80 at 7th index being number 7 as output

y_test[0]

# for metrics to be calculated, y_test values are in integer numbers so we should convert y_predict also in that format from arrayformat for analyzing.
y_predicted_labels = [np.argmax(i) for i in y_predict]
y_predicted_labels[:5]

cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

# Now adding an hidden layer to it
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)

model.evaluate(X_test_flattened,y_test)

y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

# instead of separately creating a flattened array(X_train_flattened), we can add flatten layer directly
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)

model.evaluate(X_test,y_test)

# USING CNN
# 

model1= keras.Sequential([
    # CNN
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)), #rgb value = 1 as dataset is grayscale (like MNIST) without the colors
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

     # dense layers(ANN)
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax') # we are using softmax to normalize the probability
])

model1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model1.fit(X_train, y_train, epochs=10)

model1.evaluate(X_test,y_test)

y_predicted_new = model1.predict(X_test)
y_predicted_labels_new = [np.argmax(i) for i in y_predicted_new]
cm1 = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels_new)

plt.figure(figsize = (10,7))
sn.heatmap(cm1, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

y_predicted_new = model1.predict(X_test)
y_predicted_labels_new = [np.argmax(i) for i in y_predicted]
y_predicted_labels_new[:10]

y_test[:10]




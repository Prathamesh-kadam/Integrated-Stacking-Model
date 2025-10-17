import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier

n_samples_train = X_train.shape[0]
n_channels_train = X_train.shape[3]
n_rows_train = X_train.shape[1]
n_cols_train = X_train.shape[2]
X_train_2d = X_train.reshape(n_samples_train, n_rows_train, n_cols_train, n_channels_train)

n_samples_test = X_test.shape[0]
n_channels_test = X_test.shape[3]
n_rows_test = X_test.shape[1]
n_cols_test = X_test.shape[2]
X_test_2d = X_test.reshape(n_samples_test, n_rows_test, n_cols_test, n_channels_test)
n_classes = len(np.unique(y_train))

y_train_categorical = to_categorical(y_train, num_classes=n_classes)

y_test_categorical = to_categorical(y_test, num_classes=n_classes)

def cnn():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(n_rows_train, n_cols_train, n_channels_train)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))  # Update the number of neurons here

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(cnn)

model.fit(X_train_2d, y_train_categorical, epochs=50, batch_size=32)

y_train_pred = model.predict(X_train_2d)
y_test_pred = model.predict(X_test_2d)

train_accuracy = model.score(X_train_2d, y_train_categorical)
print('Training Accuracy:', train_accuracy)

test_accuracy = model.score(X_test_2d, y_test_categorical)
print('Test Accuracy:', test_accuracy)

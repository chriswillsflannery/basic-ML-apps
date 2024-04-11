import keras
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test)= mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()
print(y_train[0])

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_train /= 255

x_test = x_test.astype('float32')
x_test /= 255

# use one-hot encoding to train model to think categorically i.e.
# 10% chance it's 0, 10% chance it's 1, 90% chance it's 2...
print(y_train[0])
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print (y_train[0])

model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=input_shape),
    keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    #dropout layer - remove useless neurons
    #probability of neuron getting dropped out is 25%
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

print(model.evaluate(x_test, y_test))
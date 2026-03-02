import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST as NumPy arrays
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

# Normalize to [0,1] and add channel dimension: (N, 28, 28, 1)
train_x = (train_x.astype("float32") / 255.0)[..., None]
test_x  = (test_x.astype("float32")  / 255.0)[..., None]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(train_x, train_y,
                    validation_split=0.1,
                    epochs=5,
                    batch_size=128,
                    verbose=2)

# Graph: training curves
plt.figure()
plt.title("MNIST training accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.legend()
plt.show()

# Save the model (recommended high-level Keras format)
model.save("digit_model.keras")
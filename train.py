import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load dataset
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

# Normalize
X_train, X_test = X_train / 255.0, X_test / 255.0

# Build model
model = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=5)

# Save model
#model.save("model/mnist_model.h5")
# model = tf.keras.models.load_model("model/mnist_model.keras")
model = tf.keras.models.load_model(
    "model/mnist_model.h5",
    compile=False
)
print("Model saved!")
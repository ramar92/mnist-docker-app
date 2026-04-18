from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from PIL import Image
import os

app = Flask(__name__)

MODEL_PATH = "model/mnist_model.keras"

# ✅ Train model inside container if not exists
def train_model():
    (X_train, y_train), _ = datasets.mnist.load_data()
    X_train = X_train / 255.0

    model = models.Sequential([
        layers.Input(shape=(28,28)),   # ✅ FIX: explicit Input layer
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=3)

    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)
    return model

# ✅ Load or train model
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = train_model()


@app.route("/")
def home():
    return "MNIST API Running (Docker Stable Version)"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]

        image = Image.open(file).convert("L").resize((28,28))
        img = np.array(image) / 255.0
        img = img.reshape(1,28,28)

        prediction = model.predict(img)
        digit = int(np.argmax(prediction))

        return jsonify({"digit": digit})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
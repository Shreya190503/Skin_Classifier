import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = 224

def load_tf_model(path):
    return tf.keras.models.load_model(str(path))


def preprocess_tf(image: Image.Image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype("float32") / 255.0

    # NHWC
    image = np.expand_dims(image, axis=0)  # (1, 224, 224, 3)
    return image


def predict_tf(model, image: Image.Image):
    x = preprocess_tf(image)
    preds = model.predict(x, verbose=0)
    return preds[0]

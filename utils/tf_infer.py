import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = 224

def load_tf_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path)

def predict_tf(model, image):
    import tensorflow as tf
    import numpy as np
    from PIL import Image

    image = image.resize((224, 224))
    x = np.array(image) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x, verbose=0)
    return preds[0]


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

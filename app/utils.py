from tensorflow.keras.preprocessing import image
import numpy as np

IMAGE_SIZE = (150, 150)

def prepare_image(img_path):
    """Preprocess the uploaded image for model prediction"""
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

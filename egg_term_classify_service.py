import cv2
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model

egg_status = ['000d', '002d', '004d', '006d', '008d', '010d', '012d', '014d', '016d', '018d', '020d', 'none']
img_size = 180
model = load_model("egg_term_model.h5")

def classify_image(image_path):
    gray_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray_test_img.jpg", gray_img)

    input_image = tf.keras.utils.load_img("gray_test_img.jpg", target_size=(img_size, img_size))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    label = egg_status[np.argmax(result)]
    accuracy = str(np.max(result) * 100) + "%"

    os.remove("gray_test_img.jpg")

    return label, accuracy
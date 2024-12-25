import os

import cv2
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential


def preprocess_image(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            img_path = os.path.join(subdir, file)
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gaussian_blur = cv2.GaussianBlur(gray_img, (0, 0), 2.0)
            sharpened = cv2.addWeighted(gray_img, 2.0, gaussian_blur, -1.0, 0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(sharpened)
            denoised = cv2.fastNlMeansDenoising(
                enhanced,
                h=10,
                templateWindowSize=7,
                searchWindowSize=21
            )
            relative_path = os.path.relpath(subdir, input_dir)
            output_subdir = os.path.join(output_dir, relative_path)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            output_path = os.path.join(output_subdir, file)
            cv2.imwrite(output_path, denoised)


input_dir = "dataset"
output_dir = "dataset_preprocessed"
preprocess_image(input_dir, output_dir)

print("----------------------------------------")
dataset_dir = os.listdir("dataset_preprocessed")
count = 0
for dir in dataset_dir:
    files = list(os.listdir("dataset_preprocessed/" + dir))
    print("Label dir:" + dir + "|" + str(len(files)) + " imgs")
    count += len(files)
print("Total: " + str(count) + " imgs")
print("----------------------------------------")

dataset_dir = "dataset_preprocessed/"
img_size = 180
batch_size = 32

training_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    seed=123,
    validation_split=0.2,
    subset="training",
    batch_size=batch_size,
    image_size=(img_size, img_size),
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    seed=123,
    validation_split=0.2,
    subset="validation",
    batch_size=batch_size,
    image_size=(img_size, img_size),
)

print("----------------------------------------")
egg_status = training_dataset.class_names
print("Labels: ")
print(egg_status)
print("----------------------------------------")

AUTOTUNE = tf.data.AUTOTUNE
training_dataset = training_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

data_augmentation = Sequential([
    layers.RandomFlip("horizontal", input_shape=(img_size, img_size, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

model = Sequential([
    data_augmentation,
    layers.Rescaling(1.0 / 255),
    Conv2D(16, 3, padding="same", activation="relu"),
    MaxPooling2D(),
    Conv2D(32, 3, padding="same", activation="relu"),
    MaxPooling2D(),
    Conv2D(64, 3, padding="same", activation="relu"),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(len(egg_status)),
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.summary()

history = model.fit(training_dataset, epochs=15, validation_data=validation_dataset)

model.save("egg_term_model.h5")

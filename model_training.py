import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def load_data(data_dir, label):
    images = []
    labels = []
    images_dir = os.path.join(data_dir, 'images')
    
    if not os.path.isdir(images_dir):
        print(f"Directory {images_dir} does not exist.")
        return np.array(images), np.array(labels)
    
    print(f"Loading {label} images from {images_dir}")
    class_num = 0 if label == 'COVID' else 1
    
    for img in os.listdir(images_dir):
        img_path = os.path.join(images_dir, img)
        if not os.path.isfile(img_path):
            print(f"File {img_path} does not exist.")
            continue
        try:
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_array is None:
                print(f"Failed to read image {img_path}.")
                continue
            img_resized = cv2.resize(img_array, (150, 150))
            images.append(img_resized)
            labels.append(class_num)
            print(f"Processed {img_path}")
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    return np.array(images), np.array(labels)

covid_dir = os.path.join(os.getcwd(), 'COVID-19_Radiography_Dataset', 'COVID')
normal_dir = os.path.join(os.getcwd(), 'COVID-19_Radiography_Dataset', 'Normal')

covid_images, covid_labels = load_data(covid_dir, 'COVID')
normal_images, normal_labels = load_data(normal_dir, 'Normal')

images = np.concatenate((covid_images, normal_images), axis=0)
labels = np.concatenate((covid_labels, normal_labels), axis=0)

if len(images) == 0 or len(labels) == 0:
    print("No images or labels loaded. Please check the dataset path and contents.")
    exit(1)

print(f"Loaded {len(images)} images.")
images = images / 255.0  # Normalize pixel values

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train = X_train.reshape(-1, 150, 150, 1)
X_test = X_test.reshape(-1, 150, 150, 1)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model.save('corona_detector_model.h5')

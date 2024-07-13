import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split

# Define image directory and target size
image_directory = r'Week 2\\homer_bart'
target_size = (64, 64)

# Load images and labels
def load_images_and_labels(image_directory, target_size):
    images = []
    labels = []  # Assuming the folder names are the labels/classes

    for label in os.listdir(image_directory):
        class_path = os.path.join(image_directory, label)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                image = load_img(image_path, target_size=target_size)
                image = img_to_array(image)
                images.append(image)
                labels.append(label)

    images = np.array(images, dtype='float32') / 255.0  # Normalize images

    label_to_int = {label: idx for idx, label in enumerate(set(labels))}
    labels = [label_to_int[label] for label in labels]

    labels = np.array(labels)
    return images, labels

images, labels = load_images_and_labels(image_directory, target_size)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=40)

# Define the model
model = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(512, activation='linear'),
    # Dropout(0.1),  # Add dropout to prevent overfitting
    Dense(256, activation='linear'),
    # Dropout(0.1),  # Add dropout to prevent overfitting
    
    Dense(1, activation='sigmoid')  # Assuming binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f'Test accuracy: {test_acc}')

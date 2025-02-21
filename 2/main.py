import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# Load CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255.0, testing_images / 255.0  # Normalize images

# Class names for CIFAR-10
class_names = ['Plane', 'Car', 'Cat', 'Bird', 'Dog', 'Deer', 'Frog', 'Horse', 'Ship', 'Truck']

# Display first 16 images
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])  # Fixed indexing issue

plt.show()

training_images, training_labels = training_images[:200000], training_labels[:200000]
testing_images, testing_labels = testing_images[:400000], testing_labels[:400000]  # Fixed from 40000 to 4000

# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),  # Fixed input shape
#     layers.MaxPooling2D((2, 2)),  # Fixed typo
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),  # Fixed typo
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Fixed typo

# # Train the model
# model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))  # Fixed validation data


# loss, accuracy = model.evaluate(testing_images, testing_labels)
# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy} ")

# model.save('image_class.keras')

model = models.load_model('image_class.keras')

img= cv.imread('h.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img,cmap=plt.cm.binary)
prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f'precition is {class_names[index]}')
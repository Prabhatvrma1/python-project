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


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
class_names = ['Plane', 'Car', 'Cat', 'Bird', 'Dog', 'Deer', 'Frog', 'Horse', 'Ship', 'Truck']
model = models.load_model('image_class.keras')
img = cv.imread('plane.webp')
if img is None:
    raise ValueError("Error: Image not found. Check the file path.")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

img = cv.resize(img, (32, 32))
img = img / 255.0
plt.imshow(img)
plt.axis("off")
plt.show()
prediction = model.predict(np.array([img]))
index = np.argmax(prediction)

print(f'Prediction is {class_names[index]}')

# Image Classification Model

## Overview
This project implements an image classification model using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained to classify images into one of 10 categories: Plane, Car, Cat, Bird, Dog, Deer, Frog, Horse, Ship, and Truck.

## Model Architecture
The model consists of the following layers:
- **Conv2D (32 filters, 3x3 kernel, ReLU activation, input shape: 32x32x3)**
- **MaxPooling2D (2x2 pool size)**
- **Conv2D (64 filters, 3x3 kernel, ReLU activation)**
- **MaxPooling2D (2x2 pool size)**
- **Conv2D (64 filters, 3x3 kernel, ReLU activation)**
- **Flatten Layer**
- **Dense Layer (64 neurons, ReLU activation)**
- **Dense Layer (10 neurons, Softmax activation)**

## Model Compilation & Training
- The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss function.
- It is trained for **10 epochs** using a dataset with **training and validation images**.
- After training, the model's accuracy and loss are evaluated using the test dataset.
- The trained model is saved as `image_class.keras`.

## Image Prediction
1. **Load the trained model**
   ```python
   from tensorflow.keras import models
   model = models.load_model('image_class.keras')
   ```
2. **Read and preprocess the input image**
   - Load the image using OpenCV.
   - Convert BGR to RGB format.
   - Resize it to 32x32 pixels.
   - Normalize pixel values between 0 and 1.
   ```python
   import cv2 as cv
   import numpy as np
   import matplotlib.pyplot as plt
   
   img = cv.imread('plane.webp')
   if img is None:
       raise ValueError("Error: Image not found. Check the file path.")
   img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
   img = cv.resize(img, (32, 32))
   img = img / 255.0
   plt.imshow(img)
   plt.axis("off")
   plt.show()
   ```
3. **Make a prediction**
   ```python
   prediction = model.predict(np.array([img]))
   index = np.argmax(prediction)
   class_names = ['Plane', 'Car', 'Cat', 'Bird', 'Dog', 'Deer', 'Frog', 'Horse', 'Ship', 'Truck']
   print(f'Prediction is {class_names[index]}')
   ```

## Requirements
- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Matplotlib

## How to Run
1. Install dependencies:
   ```sh
   pip install tensorflow opencv-python numpy matplotlib
   ```
2. Train the model and save it as `image_class.keras`.
3. Run the prediction script with an image file.

## Notes
- Ensure the image file exists in the specified path before running the script.
- The model is trained on a predefined dataset and may require retraining for different datasets.

## License
This project is open-source and can be modified or extended as needed.


import cv2
import numpy as np
from keras.models import load_model
import os

model = load_model('alexnet_model.h5')

def predict_folder(model, folder_path):
    images = []
    for image_file in os.listdir(folder_path):
        # Load image
        img = cv2.imread(os.path.join(folder_path, image_file), cv2.IMREAD_GRAYSCALE)
        # Preprocess image
        img = cv2.resize(img, (512, 512))
        img = cv2.equalizeHist(img)
        img = np.expand_dims(img, axis=-1)
        # Add image to list
        images.append(img)
    
    # Convert list of images to numpy array
    images = np.array(images, dtype=np.float32)
    
    # Make predictions
    predictions = model.predict(images)
    
    return predictions

predictions = predict_folder(model, "./train/healthy")
print(predictions)
import tensorflow as tf
import cv2
import json
import numpy as np
import os
from matplotlib import pyplot as plt
import albumentations as alb
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import accuracy_score

#facetracker = load_model('facetracker.h5')
#trained_model = load_model('model.h5', compile=False)
facetracker = load_model('smiletracker.h5', compile=False)
#img = mpimg.imread('Data/NewImages/ffb80687-f0b3-11ec-a012-9840bb2b25db.jpg')
img = cv2.imread('D:/Mihailo/SI/VIII semestar/Osnivi dubokog ucenja/Face Detection/Data/Testiranje/b3ad1e5b-f21d-11ec-bedb-9840bb2b25db.jpg', cv2.IMREAD_UNCHANGED)
#plt.imshow(img)
resized = img
#print("Ovde sam")
dim = (120, 120)
new_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
yhat = facetracker.predict(np.expand_dims(new_image/255,0))
#print(yhat[0])
sample_coords = yhat[1][0]
dim = (450, 450)
new_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#plt.imshow(new_image)
start_coordinates = tuple(np.multiply(sample_coords[:2], [450,450]).astype(int))
end_coordintes = tuple(np.multiply(sample_coords[2:], [450,450]).astype(int))
image = cv2.rectangle(new_image, start_coordinates, end_coordintes,
                      (255,0,0), 2)
#print(image)
cv2.imshow('Smile face', image)
cv2.waitKey()

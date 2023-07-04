import cv2
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

X, Y=fetch_openml('mnist_784', version=1, return_X_y=True)
classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
xtrain, xtest, ytrain, ytest= train_test_split(X, Y, random_state=42, train_size=7500, test_size=2500)
xtrainscale=xtrain/255.0
xtestscale=xtest/255.0
clf=LogisticRegression(solver='saga', multi_class='multinomial').fit(xtrainscale, ytrain)
def predictimage(image):
    im=Image.open(image)
    im1=im.convert('L')
    im2=im1.resize((28,28), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(im2, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(im2-min_pixel, 0, 255)
    max_pixel = np.max(im2)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_predict=clf.predict(test_sample)
    return test_predict[0]

p=predictimage('digit3.jpeg')
print(p)
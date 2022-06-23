import tensorflow as tf
import cv2
import json
import numpy as np
import os
from matplotlib import pyplot as plt
import albumentations as alb 

transformation = alb.Compose([alb.RandomCrop(450, 450),
                        alb.HorizontalFlip(p=0.5),
                        alb.RandomBrightnessContrast(p=0.3),
                        alb.RandomGamma(p=0.3),
                        alb.RGBShift(p=0.3),
                        alb.VerticalFlip(p=0.5)],
                        bbox_params=alb.BboxParams(format='albumentations',
                                                   label_fields=['class_names'])
                       )

augmentedDictionary = {}

for partition in ['NewTrain','NewTest','NewVal']: 
    for image in os.listdir(os.path.join('Data', partition, 'NewImages')):
        img = cv2.imread(os.path.join('Data', partition, 'NewImages', image))
        
        splitingImageName = image.split(".")[0]
        jsonName = '{}.json'.format(splitingImageName)
        coords = [0,0,0.00001,0.00001]
        label_path = os.path.join('Data', partition, 'NewLabels', jsonName)

        if os.path.exists(label_path):
            file = open(label_path, 'r')
            dataDictionary = json.load(file)
            coords[0] = dataDictionary['shapes'][0]['points'][0][0]/640
            coords[1] = dataDictionary['shapes'][0]['points'][0][1]/480
            coords[2] = dataDictionary['shapes'][0]['points'][1][0]/640
            coords[3] = dataDictionary['shapes'][0]['points'][1][1]/480
            
        try:
            for augmentedImageNumber in range(20):
                transformed = transformation(image=img, bboxes=[coords], class_names=['Smile'])
                splitingImageName = image.split(".")[0]
                jpgName = '{}.{}.jpg'.format(splitingImageName, augmentedImageNumber)
                cv2.imwrite(os.path.join('Aug_data', partition, 'NewImagess', jpgName), transformed['image'])
                augmentedDictionary['image'] = image
                if len(transformed['bboxes']) == 0: 
                    augmentedDictionary['bbox'] = [0,0,0,0]
                    augmentedDictionary['class'] = 0
                else:
                    augmentedDictionary['bbox'] = transformed['bboxes'][0]
                    augmentedDictionary['class'] = 1
                jsonName = '{}.{}.json'.format(splitingImageName, augmentedImageNumber)
                with open(os.path.join('Aug_data', partition, 'NewLabelss', jsonName), 'w') as jsonFile:
                    json.dump(augmentedDictionary, jsonFile)
        except Exception as e:
            print(e)
import tensorflow as tf
import cv2
import json
import numpy as np
import os
from matplotlib import pyplot as plt
import albumentations as alb
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16, ResNet50V2
from tensorflow.keras.models import load_model


def load_image(image):
    byte_image = tf.io.read_file(image)
    img = tf.io.decode_jpeg(byte_image)
    return img

def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
        
    return [label['class']], label['bbox']

def localization_loss(y_true, yhat):            
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
                  
    h_true = y_true[:,3] - y_true[:,1] 
    w_true = y_true[:,2] - y_true[:,0] 

    h_pred = yhat[:,3] - yhat[:,1] 
    w_pred = yhat[:,2] - yhat[:,0] 
    
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
    
    return delta_coord + delta_size

def build_model(): 
    input_layer = Input(shape=(120,120,3))
    
    vgg = VGG16(include_top=False)(input_layer)

    # Classification Model  
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)
    
    # Bounding box model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)
    
    smiletracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return smiletracker

class SmileTracker(Model): 
    def __init__(self, smiletracker,  **kwargs): 
        super().__init__(**kwargs)
        self.model = smiletracker

    def compile(self, opt, classification_loss, localization_loss, **kwargs):
        super().compile(**kwargs)
        self.closs = classification_loss
        self.lloss = localization_loss
        self.opt = opt
    
    def train_step(self, data, **kwargs): 
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        X, y = data
        
        with tf.GradientTape() as tape: 
            classes, coordinates = self.model(X, training=True) # Forward pass
            
            classloss = self.closs(y[0], classes)
            localizationloss = self.lloss(tf.cast(y[1], tf.float32), coordinates)
            
            total_loss = localizationloss+0.7*classloss
            # Compute gradients
            trainable_vars = self.trainable_variables
            grad = tape.gradient(total_loss, trainable_vars)
        # Update weights
        opt.apply_gradients(zip(grad, trainable_vars))
        # Return a dict mapping losses
        return {"Total_loss":total_loss, "Class_loss": classloss, "Regress_loss": localizationloss}
        
    def test_step(self, data, **kwargs): 
        # Unpack the data
        X, y = data
        # Compute predictions
        classes, coordinates = self.model(X, training=False)
        
        classloss = self.closs(y[0], classes)
        localizationloss = self.lloss(tf.cast(y[1], tf.float32), coordinates)
        total_loss = localizationloss+0.7*classloss
        # Return a dict mapping losses of predictions
        return {"Total_loss":total_loss, "Class_loss": classloss, "Regress_loss": localizationloss}
        
    def call(self, X, **kwargs): 
        return self.model(X, **kwargs)

train_images = tf.data.Dataset.list_files('Aug_data\\NewTrain\\NewImages\\*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120,120)))
train_images = train_images.map(lambda x: x/255)
test_images = tf.data.Dataset.list_files('Aug_data\\NewTest\\NewImages\\*.jpg', shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (120,120)))
test_images = test_images.map(lambda x: x/255)
val_images = tf.data.Dataset.list_files('Aug_data\\NewVal\\NewImages\\*.jpg', shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (120,120)))
val_images = val_images.map(lambda x: x/255)

train_labels = tf.data.Dataset.list_files('Aug_data\\NewTrain\\NewLabels\\*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
test_labels = tf.data.Dataset.list_files('Aug_data\\NewTest\\Newlabels\\*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
val_labels = tf.data.Dataset.list_files('Aug_data\\NewVal\\NewLabels\\*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(5000)
train = train.batch(8)
train = train.prefetch(4)
test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1000)
test = test.batch(8)
test = test.prefetch(4)
val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(4)

#vgg = VGG16(include_top=False)
#resNet = ResNet50V2(include_top=False)
#resNet.summary()


smileTracker = build_model()
#facetracker.summary()

beches = len(train)
lr = (1./0.75 -1)/beches
opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr)

classification_loss = tf.keras.losses.BinaryCrossentropy()
regression_loss = localization_loss

model = SmileTracker(smileTracker)
model.compile(opt, classification_loss, regression_loss)

callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
hist = model.fit(train, epochs=10, validation_data=val, callbacks=[callback])

'''
fig, ax = plt.subplots(ncols=3, figsize=(20,5))

ax[0].plot(hist.history['total_loss'], color='blue', label='loss')
ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
ax[0].title.set_text('Loss')
ax[0].legend()

ax[1].plot(hist.history['class_loss'], color='blue', label='class loss')
ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
ax[1].title.set_text('Classification Loss')
ax[1].legend()

ax[2].plot(hist.history['regress_loss'], color='blue', label='regress loss')
ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
ax[2].title.set_text('Regression Loss')
ax[2].legend()

plt.show()
'''
#smileTracker.save('smiletrackerNewModel.h5')

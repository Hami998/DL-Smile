# -*- coding: utf-8 -*-
"""
Created on Wed May 11 00:14:20 2022

@author: user
"""

#import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.regularizers import Regularizer, l2
#from tensorflow.keras.layers import Dense
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
from sklearn.neural_network import MLPClassifier, MLPRegressor

"Load dataset"
dataFrame = pd.read_csv("dataset.csv")

columns = list(dataFrame.columns)
print(columns)

for col in columns:
   dataFrame[col] = dataFrame[col].replace("?", np.nan)
   dataFrame[col] = dataFrame[col].astype('float32')
   dataFrame[col].fillna(dataFrame[col].median(), inplace=True) 
# za proveru koliko ima vrednosti koje nedostaju
#print(dataFrame.isna().sum())

for i in range(0, len(dataFrame['Income'])):
    if dataFrame['Income'][i] == 1 or dataFrame['Income'][i] == 2:
        dataFrame['Income'][i] = 0
    elif dataFrame['Income'][i] == 3 or dataFrame['Income'][i] == 4:
        dataFrame['Income'][i] = 1
    elif dataFrame['Income'][i] == 5 or dataFrame['Income'][i] == 6:
        dataFrame['Income'][i] = 2
    elif dataFrame['Income'][i] == 7 or dataFrame['Income'][i] == 8:
        dataFrame['Income'][i] = 3
    elif dataFrame['Income'][i] == 9:
        dataFrame['Income'][i] = 4
        

'''
dataFrame["Sex"] = dataFrame["Sex"].replace("?", 1)
dataFrame["MaritalStatus"] = dataFrame["MaritalStatus"].replace("?", 3)
dataFrame["Age"] = dataFrame["Age"].replace("?", 4)
dataFrame["Education"] = dataFrame["Education"].replace("?", 3)
dataFrame["Occupation"] = dataFrame["Occupation"].replace("?", 5)
dataFrame["YearsInSf"] = dataFrame["YearsInSf"].replace("?", 3)
dataFrame["DualIncome"] = dataFrame["DualIncome"].replace("?", 2)
dataFrame["HouseholdMembers"] = dataFrame["HouseholdMembers"].replace("?", 5)
dataFrame["Under18"] = dataFrame["Under18"].replace("?", 5)
dataFrame["HouseholdStatus"] = dataFrame["HouseholdStatus"].replace("?", 2)
dataFrame["TypeOfHome"] = dataFrame["TypeOfHome"].replace("?", 3)
dataFrame["EthnicClass"] = dataFrame["EthnicClass"].replace("?", 4)
dataFrame["Language"] = dataFrame["Language"].replace("?", 1)
dataFrame["Sex"] = dataFrame["Sex"].astype(float)
dataFrame["MaritalStatus"] = dataFrame["MaritalStatus"].astype(float)
dataFrame["Age"] = dataFrame["Age"].astype(float)
dataFrame["Education"] = dataFrame["Education"].astype(float)
dataFrame["Occupation"] = dataFrame["Occupation"].astype(float)
dataFrame["YearsInSf"] = dataFrame["YearsInSf"].astype(float)
dataFrame["DualIncome"] = dataFrame["DualIncome"].astype(float)
dataFrame["HouseholdMembers"] = dataFrame["HouseholdMembers"].astype(float)
dataFrame["Under18"] = dataFrame["Under18"].astype(float)
dataFrame["HouseholdStatus"] = dataFrame["HouseholdStatus"].astype(float)
dataFrame["TypeOfHome"] = dataFrame["TypeOfHome"].astype(float)
dataFrame["EthnicClass"] = dataFrame["EthnicClass"].astype(float)
dataFrame["Language"] = dataFrame["Language"].astype(float)
dataFrame["Income"] = dataFrame["Income"].astype(float)

dataFrame["Sex"] = dataFrame["Sex"] / dataFrame["Sex"].max()
dataFrame["MaritalStatus"] = dataFrame["MaritalStatus"] / dataFrame["MaritalStatus"].max()
dataFrame["Age"] = dataFrame["Age"] / dataFrame["Age"].max()
dataFrame["Education"] = dataFrame["Education"] / dataFrame["Education"].max()
dataFrame["Occupation"] = dataFrame["Occupation"] / dataFrame["Occupation"].max()
dataFrame["YearsInSf"] = dataFrame["YearsInSf"] / dataFrame["YearsInSf"].max()
dataFrame["DualIncome"] = dataFrame["DualIncome"] / dataFrame["DualIncome"].max()
dataFrame["HouseholdMembers"] = dataFrame["HouseholdMembers"] / dataFrame["HouseholdMembers"].max()
dataFrame["Under18"] = dataFrame["Under18"] / dataFrame["Under18"].max()
dataFrame["HouseholdStatus"] = dataFrame["HouseholdStatus"] / dataFrame["HouseholdStatus"].max()
dataFrame["TypeOfHome"] = dataFrame["TypeOfHome"] / dataFrame["TypeOfHome"].max()
dataFrame["EthnicClass"] = dataFrame["EthnicClass"] / dataFrame["EthnicClass"].max()
dataFrame["Language"] = dataFrame["Language"] / dataFrame["Language"].max()

'''

##our target
##target = dataFrame.pop('Income')

dataFrame = dataFrame.sample(frac=1, random_state=10)

X = dataFrame[columns[0:13]].values
y = dataFrame[columns[13]].values
#y=y.astype('int')
#scaler = StandardScaler()
#X = pd.DataFrame(scaler.fit_transform(X))
#y = pd.DataFrame(scaler.fit_transform(y))

#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.4)
#y=y.reshape(y.shape[0],1)
#print(y)
'''
mlp = MLPClassifier(hidden_layer_sizes=(128,64,64,32,5), activation='relu', solver='adam', max_iter=5000)
mlp.fit(x_train,y_train)

p=mlp.predict(x_test)
#for i in range(len(p)):
    #print(x_test[i][30], x_test[i][31], y_test[i], p[i])

confussionMatrix = confusion_matrix(y_test, p)

print(accuracy_score(y_test, p))
print(confusion_matrix(y_test, p))
'''

#,  random_state=10, stratify=y
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.30)
X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, test_size= 0.30, random_state=10, stratify=y)
x_valid, x_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, test_size= 0.50, random_state=10, stratify=y_valid_test)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
x_valid = pd.DataFrame(scaler.fit_transform(x_valid))
X_test = pd.DataFrame(scaler.fit_transform(x_test))


#X_test = pd.DataFrame(scaler.fit_transform(X_test))


#sm =SMOTE(random_state=2)
#X_train_res, y_train_res = sm.fit_resample(X_train, y_train)


'''
#Sluzi kao provera raspodele zarade, mozda treba sampling da se doda
counter = 0
for i in range(0, 10):
    for j in range(0, len(y_train)):
        if(y_train_res[j] == i):
            counter += 1
    print("Za broj: ")
    print(i)
    print("===========")
    print(counter)
    counter = 0

'''
'''
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
'''

'''
Provera za SVM
model_SVC = SVC()
model_SVC.fit(X_train, y_train)
y_pred = model_SVC.predict(X_test)
print(accuracy_score(y_test, y_pred))
'''

model = keras.Sequential([
    keras.layers.Dense(128, input_dim=13, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(9, activation='relu'),
    keras.layers.Dense(9, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(5, activation='softmax'),
    ])

'''
model = keras.Sequential([
    keras.layers.Dense(26, input_dim=13, activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    #keras.layers.Dense(64, activation='relu'),
    #keras.layers.Dropout(0.3),
    #keras.layers.Dense(9, activation='relu'),
    #keras.layers.Dense(9, activation='relu'),
    #keras.layers.Dropout(0.3),
    keras.layers.Dense(5, activation='softmax'),
    ])
'''
#kernel_regularizer=tf.keras.regularizers.L1(0.01), activity_regularizer=tf.keras.regularizers.L2(0.01)
model.compile(
    loss='categorical_crossentropy',
    #loss='mse',
    optimizer='adam',
    metrics=['accuracy']
    )

y_train_enc = pd.get_dummies(y_train)
y_valid_enc = pd.get_dummies(y_valid)

stopping = EarlyStopping(monitor='val_loss', patience=25)

history = model.fit(X_train, y_train_enc, validation_data = (x_valid,y_valid_enc), epochs = 200, callbacks=stopping)
#history = model.fit(X_train, y_train, epochs = 500, batch_size=128)
#batch_size=128
#, validation_split = 0.2
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
'''
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
'''
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1)

y_pred_enc = pd.get_dummies(y_pred)
y_test_enc = pd.get_dummies(y_test)

target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']

##stampaj y_pred i y_test
## y_pred ide od 0 do 4
## test ide od 1 do 5
## tu je greska
#y_pred = dict(enumerate(y_pred))
#y_test = dict(enumerate(y_test))

#print("{}".format(y_pred))
#print("===================")
#print("{}".format(y_test))

print("Model accuracy score is: {}".format(accuracy_score(y_test_enc, y_pred_enc)))
print(classification_report(y_test_enc, y_pred_enc, target_names=target_names))
#print(precision_recall_fscore_support(y_test_enc, y_pred_enc))
matrix = confusion_matrix(y_test, y_pred)
print(matrix)
print(matrix.diagonal()/matrix.sum(axis=1))

Accuracy = matrix.diagonal()/matrix.sum(axis=1)
Accuracy = dict(enumerate(Accuracy))
Accuracy['class 0'] = Accuracy.pop(0)
Accuracy['class 1'] = Accuracy.pop(1)
Accuracy['class 2'] = Accuracy.pop(2)
Accuracy['class 3'] = Accuracy.pop(3)
Accuracy['class 4'] = Accuracy.pop(4)
print(Accuracy)
#print(accuracy_score(y_test, y_pred))

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
'''
sklearn_weights = class_weight.compute_class_weight("balanced", np.unique(y_train), y_train)
print(sklearn_weights)

sklearn_weights = dict(enumerate(sklearn_weights))


model = keras.Sequential([
    keras.layers.Dense(128, input_dim=13, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(62, activation='relu'),
    #keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    #keras.layers.Dropout(0.3),
    keras.layers.Dense(5, activation='softmax'),
    ])

model.compile(
    loss='categorical_crossentropy',
    #loss='mse',
    optimizer='adam',
    metrics=['accuracy']
    )

y_train_enc = pd.get_dummies(y_train)
y_valid_enc = pd.get_dummies(y_valid)

history = model.fit(X_train, y_train_enc, validation_data = (x_valid,y_valid_enc), epochs = 50, class_weight=sklearn_weights)
#history = model.fit(X_train, y_train, epochs = 500, batch_size=128)
#batch_size=128
#, validation_split = 0.2
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1)

y_pred_enc = pd.get_dummies(y_pred)
y_test_enc = pd.get_dummies(y_test)

print(accuracy_score(y_test_enc, y_pred_enc))

'''

'''

loss_test, acc_test = model.evaluate(X_test, y_test, verbose=1)
print(loss_test)
print(acc_test)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))

#print(accuracy_score(y_test, max(y_pred)))
'''

'''
model = Sequential()
model.add(Dense(40, input_dim=13, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))
# compile the keras model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=10)

'''

'''

#model = DecisionTreeClassifier() 
#model = RandomForestClassifier(n_estimators=10)
model = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
#print(accuracy_score(y_train, y_pred))
print(confusion_matrix(y_test, y_pred))

'''

'''
ArrayX_train = np.array(X_train)
ArrayX_test = np.array(X_test)
ArrayY_train = np.array(y_train)
ArrayY_test = np.array(y_test)

train_X = tf.convert_to_tensor(ArrayX_train)
test_X = tf.convert_to_tensor(ArrayX_test)
train_y = tf.convert_to_tensor(ArrayY_train)
test_y = tf.convert_to_tensor(ArrayY_test)

## converting dataFrame to np.array 
## because tensorflow accepts it
##dataArray = np.array(dataFrame)
##print(tf.convert_to_tensor(dataArray))
#model = tf.keras.Sequential([
#    tf.keras.layers.Flatten(input_shape=(13,)),
#    tf.keras.layers.Dense(128, activation='relu'),
#    tf.keras.layers.Dense(1)
#])

model = tf.keras.models.Sequential([
    #tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dense(32,activation="relu"),
    tf.keras.layers.Dense(9,activation="softmax")])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
   optimizer=tf.keras.optimizers.Adam(),
   metrics="accuracy")

#model.compile(optimizer='adam',
#                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#                metrics=['accuracy'])
#model.fit(train_X, train_y, epochs = 100)
model.fit(train_X, train_y, epochs = 100)
print("===================")
#model.evaluate(test_X, test_y)
model.evaluate(test_X, test_y)

#print(tf.convert_to_tensor([[1, 2], [3, 4]]))
'''





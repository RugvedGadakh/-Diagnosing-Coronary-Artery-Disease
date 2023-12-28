#!/usr/bin/env python
# coding: utf-8

# # Diagnosing Coronary Artery Disease 
# 

# **Data Set Information:**
# 
# This dataset contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4. Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0).
# 
# The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.
# 
# One file has been "processed", that one containing the Cleveland database. All four unprocessed files also exist in this directory.

# **Attribute Information:**
# 
# Only 14 attributes used:
# 1. #3 (age)
# 2. #4 (sex)
# 3. #9 (cp)
# 4. #10 (trestbps)
# 5. #12 (chol)
# 6. #16 (fbs)
# 7. #19 (restecg)
# 8. #32 (thalach)
# 9. #38 (exang)
# 10. #40 (oldpeak)
# 11. #41 (slope)
# 12. #44 (ca)
# 13. #51 (thal)
# 14. #58 (num) (the predicted attribute)

# ## Import Libraries

# In[20]:


import sys

# Data Science Tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Machine Learning Tools
import sklearn
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
import pickle

import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam



# ## Loading data

# In[2]:


# Import the heart disease dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# the names will be the names of each column in our pandas DataFrame
names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeeak", "slope", "ca", "thal", "class"]


# In[3]:


# read the csv
cleveland_data = pd.read_csv(url, names = names)


# In[4]:


cleveland_data.head()

# In[5]:


# print the shape of the Dataframe, so we can see how many examples we have

print("Shape of Dataframe: {}".format(cleveland_data.shape))
print(cleveland_data.loc[1])


# In[6]:


# print the lsat twenty or so data points
cleveland_data.loc[280:]


# ## Data Preparation

# In[7]:


# remove missing data (indicated with a "?")
data = cleveland_data[~cleveland_data.isin(['?'])]
data.loc[280:]


# In[8]:


# drop rows with NaN values from DataFrame
data = data.dropna(axis=0)
data.loc[280:]


# In[9]:


# print the shape and data type of the dataframe
print(data.shape)
print(data.dtypes)


# In[10]:


# transform data to numeric to enable further analysis
data = data.apply(pd.to_numeric)
data.dtypes


# In[11]:


# print data characteristics, usings pandas built-in describe() function
data.describe()


# In[12]:


# plot histogram for each variables
data.hist(figsize=(12, 12))
plt.show()


# ## Data Splitting (Train / Test)
# 
# Now that we have preprocessed the data appropriately, we can split it into training and testings datasets. We will use Sklearn's train_test_split() function to generate a training dataset (80 percent of the total data) and testing dataset (20 percent of the total data). 
# 
# Furthermore, the class values in this dataset contain multiple types of heart disease with values ranging from 0 (healthy) to 4 (severe heart disease). Consequently, we will need to convert our class data to categorical labels. For example, the label 2 will become [0, 0, 1, 0, 0].

# In[13]:


# create X and y datasets for training

X = np.array(data.drop("class", axis=1))

y = np.array(data["class"])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)


# In[14]:


# convert the data to categorical labels

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes = None)
print(Y_train.shape)
print(Y_train[:10])


# ## Modelling

# In[15]:


# define a function to build the keras model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim = 13, kernel_initializer = "normal", activation= "relu"))
    model.add(Dense(4, kernel_initializer= "normal", activation = "relu"))
    model.add(Dense(5, activation = "softmax"))

    # compile model
    adam = Adam(learning_rate = 0.001)
    model.compile(optimizer = adam, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

model = create_model()

print(model.summary())


# In[16]:


# fit the model to the training data
model.fit(X_train, Y_train, epochs=100, batch_size=10, verbose=1)


# In[17]:


# convert into binary classification problem  - heart disease or no heart disease
Y_train_binary = y_train.copy()
Y_test_binary = y_test.copy()

Y_train_binary[Y_train_binary > 0] = 1
Y_test_binary[Y_test_binary > 0] = 1

print(Y_train_binary[:20])


# In[18]:


# define a new keras model for binary classification
def create_binary_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    return model


binary_model = create_binary_model()

print(binary_model.summary())


# In[19]:


# fit the binary model on the training data
binary_model.fit(X_train, Y_train_binary, epochs=100, batch_size=10, verbose=1)


# In[21]:


# generate classification report using predictions for categorical model

categorical_pred = np.argmax(model.predict(X_test), axis=1)

print('Results for Categorical Model')
print(accuracy_score(y_test, categorical_pred))
print(classification_report(y_test, categorical_pred))


# In[22]:


# generate classification report using predictions for binary model
binary_pred = np.round(binary_model.predict(X_test)).astype(int)

print('Results for Binary Model')
print(accuracy_score(Y_test_binary, binary_pred))
print(classification_report(Y_test_binary, binary_pred))


# Save the binary model
pickle.dump(binary_model, open('binary_model.pkl', 'wb'))

# Load the binary model
loaded_binary_model = pickle.load(open('binary_model.pkl', 'rb'))

pickle.dump(categorical_pred, open('categorical_model.pkl', 'wb'))
loaded_categorical_model = pickle.load(open('categorical_model.pkl', 'rb'))


# ...

# Save the binary model using Keras save method
binary_model.save('binary_model.h5')

# Load the binary model using Keras load_model method
loaded_binary_model = load_model('binary_model.h5')

# Save the categorical model
model.save('categorical_model.h5')

# Load the categorical model
loaded_categorical_model = load_model('categorical_model.h5')

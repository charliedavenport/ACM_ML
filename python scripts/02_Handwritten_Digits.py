
# coding: utf-8

# # Recognizing Handwritten Digits using Scikit-Learn
# 
# MNIST dataset: https://en.wikipedia.org/wiki/MNIST_database
# 

# * 28x28 (784 pixels total) images of handwritten digits
# * each image has a label 0-9
# * 70,000 total images & labels

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt     # pretty pictures
from keras.datasets import mnist    # our dataset
from IPython.display import display # pretty tables

# The load_data() method provided here separates data and labels, and does the test/train split for us. Convenient!
(X_train, y_train), (X_test, y_test) = mnist.load_data()


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Alternative dataset source. 8x8 images rather than 28x28

# from sklearn.datasets import load_digits
# digits = load_digits()


# digits.data.shape

# plt.imshow(digits.data[0].reshape((8,8)), cmap="Greys")
# plt.show()


# # Preprocess the Data
# 
# Our data is not in the exact format we want. We will use the reshape() method from numpy to fix this transform our data from 3D arrays to 2D

# we want to preserve the # of rows, but collapse the last two dimensions into one
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
print(X_train.shape)
print(X_test.shape)


# # Display A Random Training Image
rand_index = np.random.randint(0, X_train.shape[0])
print('Showing image ', rand_index, ' from training set')
plt.imshow(X_train[rand_index].reshape(28,28), cmap='Greys') # reshape the image back to a 2d array for display
# cmap keyword: defines colorspace of image
# https://matplotlib.org/examples/color/colormaps_reference.html
#plt.axis('off')
print('Label: ', y_train[rand_index])


# # Binary Classifier
# 
# Easy starting point. Output is True or False (in the class, or not in the class)
# 
# In this case, we'll recognize whether a digit is 5.

# First, we need to restructure our labels (no need to modify the data itself)
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# let's take a peak at our new training labels
print(y_train[0:10])   # multiclass
print(y_train_5[0:10]) # binary
# now, instead of digits 0-9, we have 0's for 'not 5' and 1's for '5' - Perfect!


# # Scikit-Learn
# 
# * list of classification models: http://scikit-learn.org/stable/supervised_learning.html
# * To start, we will use a SGDClassifier http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
#     - SGD is the name of the training algorithm, "Stochastic Gradient Descent." We'll learn much more about Gradient Descent soon!
#     - This is a linear model, meaning it finds linear boundaries between classes (remember, our classes to start out with are 'five' and 'not five')
#     - The main reason I've selected this model is that it's simplicity allows for much faster training on our high-dimensional dataset. The trade-off is that it will be less accurate than more complex models.
# * To train models in sklearn, we only have to call the fit() function and pass in our data and labels

from sklearn.linear_model import SGDClassifier

# there are many parameters we can set here, but they all have default values, so we can safely leave this blank for now
classifier = SGDClassifier() 
classifier.fit(X_train, y_train_5) # This line alone trains the model. We need to pass in the data and labels
# We'll implement our own fit() function soon!

# ignore the warning - the default parameters for this class work fine for our purposes


# # Making a Prediction

rand_index = np.random.randint(0, 5000)
# once the model has been trained, we only have to call the predict() function!
prediction = classifier.predict([X_test[rand_index]]) # pass in a random row from X_test

print("Index ", rand_index, ":")
print("Predicted Label: ", prediction)
print("Actual Label (binary): ", y_test_5[rand_index])
print("Actual Label (multi-class): ", y_test[rand_index])

plt.imshow(X_test[rand_index].reshape(28,28), cmap='Greys')
plt.axis('off')
plt.show()

# # Model Evaluation

from sklearn.metrics import accuracy_score

y_test_5_pred = classifier.predict(X_test) # pass in the entire test set
score = accuracy_score(y_test_5, y_test_5_pred)
print(score)


# Wow, that's a great score! Guess our simple model did pretty well on it's own? Not necessarily.
# 
# Just because we are doing a binary classification task doesn't mean that our data is split evenly into two classes! In this case, only 10% of the samples are in the 'five' class, and the rest are 'not five', so a "dumb" model that just spits out [ False ] Every time would have *90% accuracy!*
# 
# This kind of situation is called *Label Imbalance*


# # Confusion Matrix
from sklearn.metrics import confusion_matrix
conf_mx = confusion_matrix(y_test_5, y_test_5_pred)
print(conf_mx)


# # Precision and Recall
# 
# Two more useful values we can extract from the confusion matrix are *Precision* and *Recall.*
# 
# https://en.wikipedia.org/wiki/Precision_and_recall
# 
# * Precision is the proportion of instances that were correctly labeled as 'True' out of all the 'True' predictions
#     - How many of our selected items were classified correctly?
#     - Higher precision => fewer false positives
# * Recall is the proportion of instances that were correctly labeled as 'True' out of all the instances that actually are 'True'
#     - How many of the total relevant items did we get?
#     - Higher recall => fewer false negatives
#     
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_test_5, y_test_5_pred)
recall = recall_score(y_test_5, y_test_5_pred)
print("Precision: ", precision)
print("Recall: ", recall) 


# # Multiclass Classification


multi_clf = SGDClassifier()
multi_clf.fit(X_train, y_train) # same data, different labels


# # Accuracy, Confusion Matrix
y_test_pred = multi_clf.predict(X_test)

acc = accuracy_score(y_test, y_test_pred)
conf_mx = confusion_matrix(y_test, y_test_pred) # what does the confusion matrix look like for multiclass predictions?

print(acc) # How is this accuracy score different from the binary accuracy score? 
           # Is it a better or worse measure of performance?
display(pd.DataFrame(conf_mx)) # convert to dataframe so it prints with indices, use display() to make it pretty



# Show color map for reference
plt.imshow(np.arange(0, 1, 0.01).reshape(1,-1), cmap = 'gnuplot')
plt.axis('off')
plt.show()

# Display confusion matrix as an image
plt.imshow(conf_mx, cmap = 'gnuplot')
plt.show()


# # Precision and Recall for Multiclass Predictions

precision_weighted = precision_score(y_test, y_test_pred, average="weighted")
recall_weighted = recall_score(y_test, y_test_pred, average="weighted")
precision_macro = precision_score(y_test, y_test_pred, average="macro")
recall_macro = recall_score(y_test, y_test_pred, average="macro")

print("Precision (weighted avg): ", precision_weighted)
print("Recall (weighted avg): ", recall_weighted)
print("Precision (simple avg): ", precision_macro)
print("Recall (simple avg): ", recall_macro)


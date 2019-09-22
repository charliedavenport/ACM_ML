
# coding: utf-8

# # Welcome to the ACM Machine Learning Subcommittee! 
# 
# Python libraries we will be using:
# 
# * Numpy 
#     - store and manipulate numerical data efficiently
# * Scikit-Learn 
#     - training and evaluating models
#     - Fetching datasets
# * Matplotlib 
#      - pretty pictures
#      - pyplot - MATLAB-like syntax in python

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt


# Python 3 Basics

# list: similar to an array in other languages, but can store different types
x = []
# append() method to insert at the end. insert() method to insert at an index
x.append(1)
x.append(3.14)
x.append("hello")
print(x)
# assign x to a different list object
y = ['cats', 'dogs', 'birds']
print(y)
# extend method to join two lists
x.extend(y)
print(x)

# More advanced: List Comprehensions
# equivalent to creating an empty list and populating in a for-loop
z = [i*i for i in range(10)]
print(z)


# List Slicing

# syntax: array[start:stop]
print(z)
print(z[1:5])
print(z[4:9])
# start/stop are optional. Beginning/End of the array assumed
print()
print(z[:]) # full array
print(z[3:]) # 3rd index -> end of array
print(z[:4]) # first 4 items


#  Dictionaries { }
# 
#  Store (key,value) pairs

inventory = {'carrots' : 10,
             'tomatoes' : 5,
             'bananas' : 13}

print(inventory)

# add a key,value pair
inventory['apples'] = 7
# modify a key's value
inventory['tomatoes'] = 6

# remove an entry
del inventory['carrots']

# loop through entries
for key, value in inventory.items():
    print("We have {} {}".format(value, key))


# Numpy 
# 
# Main data structure: ndarray - Multidimensional array of same numeric type
# 
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html



zero_matrix = np.zeros(shape=(3,3))
ones_matrix = np.ones(shape=(3,5))
rand_matrix = np.random.rand(5,2)

print(zero_matrix, zero_matrix.shape)
print(ones_matrix, ones_matrix.shape)
print(rand_matrix, rand_matrix.shape)

mult_matrix = np.dot(ones_matrix, rand_matrix)  # dot product == matrix-matrix multiply, matrix-vector multiply

print(mult_matrix, mult_matrix.shape) # (3 x 5) * (5 x 2) => (3 x 2)


# Array slicing
# 
# Shorthand syntax for accessing sub-arrays by 'slicing' along the array's dimensions
# 
# Suppose we only want rows 3 (inclusive) to 5 (exclusive) and columns 4 to 7. We would use the following line
# 
#     array[3:5, 4:7]

# array[start : stop]
# array[x_start : x_stop, y_start : y_stop, ...]

# if start and stop are not given, then the beginning 
# and end of that array (or that array's dimensions) are assumed

print(zero_matrix[:,:]) # the full matrix
print()
print(zero_matrix[2,:]) # just the bottom row
print()
print(ones_matrix[0, 2:5]) # row 0, columns 2,3,4 => shape=(1,3)
print()
print(rand_matrix[:3, 0:]) # rows 0,1,2, columns 0,1 => shape=(3,2)


# # Matplotlib
# 
# * Matplotlib.pyplot  -  MATLAB-like plotting syntax https://matplotlib.org/api/pyplot_api.html
# 
# * We give pyplot numpy arrays, and it plots them
# 
# 
x = np.arange(0, 5, 0.1) # another useful numpy function - gives us a 1-D array from 0 to 5, step-size=0.1
y = np.sin(x) # Pass in an array of input values, and get an array of the same shape
print(x.shape, y.shape)
plt.plot(x,y)
plt.show()

plt.scatter(rand_matrix[:,0], rand_matrix[:,1])
plt.show()

# plot options
y1 = np.cos(x)
y2 = np.tanh(x)
plt.plot(x, y2, 'm|' , label='tanh')
plt.plot(x, y1, 'g+', label='cos')
plt.plot(x, y, 'r--', label='sin')
plt.legend()
plt.show()

# We'll be using this function to visualize image data - very handy!
plt.imshow(rand_matrix, cmap='Greys')
plt.show()


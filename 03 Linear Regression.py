# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

X = 2 * np.random.rand(100, 1) # pass in desired shape. In this case, (100,1)
y = 4 + 3 * X + np.random.randn(100, 1) 
# NOTE: I'm using two different random functions. rand() gives a uniform distribution, randn() gives a Gaussian distribution


plt.scatter(X, y, s=20)
plt.axis(ymin=0.0)
plt.show()


# # Linear Regression

X_b = np.c_[np.ones((100,1)), X] # add the bias term to every instance
print(X.shape)
print(X_b.shape)


# # Selecting a Cost Function
# 
# Before we start training our model, we need to define an error function to minimize. 
# One of the most common cost functions used in regression is the Mean Squared Error, or MSE.
# 


def MSE_error(theta):
    err = 0
    m = X.shape[0]
    for i in range(m):
        y_pred = theta.T.dot(X_b[i])[0]
        err += (y_pred - y[i])**2
    return err/m


# # Training Algorithm: Gradient Descent
# 
# * https://en.wikipedia.org/wiki/Gradient_descent
# * Generic Optimization Algorithm - used in many domains outside of ML
# 
# A common analogy to explain Gradient Descent is to image that you are standing on hilly terrain covered in fog. 
# You want to find the deepest valley (minimum), but can only see your immediate surroundings. 
# Your best strategy is to look at the steepness of the terrain at your feet. Find the direction of steepest descent, 
# walk some distance in that direction, and repeat. Eventually you will find yourself in a valley.
# 
# We are trying to find the parameter values that minimize our cost function (MSE). To do this, we approximate the derivative of 
# the cost function with respect to our model parameters. When we have a multivariate (more than one input variable) function, 
# we need to compute the *Gradient* of the function. The Gradient is the multivariable generalization of the derivative. 
# It is a vector containing *partial derivatives.* 
# A partial derivative is obtained by holding all other input variables constant, and differentiating our function with respect to just one variable. 
# The Gradient, then, is a vector containing all the partial derivatives. In this case, our gradient vector will have two components.
# 

learn_rate = 0.1
n_iter = 20
m = X.shape[0]

MSE_list = np.zeros(shape=(n_iter,))

theta = np.random.randn(2,1) # random initialization of theta
print('starting value of theta:\n', theta)

plt.scatter(X, y, s=20)
for iter in range(n_iter):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y) # compute the gradient
    theta = theta - learn_rate * gradients # update theta
    
    y_0 = theta.T.dot((1,0)) # should be roughly 4
    y_2 = theta.T.dot((1,2)) # should be roughly 10
    plt.plot([0,2], [y_0,y_2])
    MSE_list[iter] = MSE_error(theta)
    
plt.show()
    
print('optimized value of theta:\n', theta)

print('MSE error at each iteration:')
plt.plot(np.arange(n_iter), MSE_list)
plt.show()


# In[6]:


y_0 = theta.T.dot((1,0)) # should be roughly 4
y_2 = theta.T.dot((1,2)) # should be roughly 10

plt.scatter(X, y, s=20)
plt.plot([0,2], [y_0,y_2])
plt.show()


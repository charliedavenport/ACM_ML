#!/usr/bin/env python
# coding: utf-8

# In[59]:


import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# In[2]:


breast_cancer = pd.read_csv('breast_cancer.csv', index_col='id')
display(breast_cancer.head())

bc_data = breast_cancer.drop(columns=['diagnosis']).dropna(axis='columns').values
bc_data_scaled = StandardScaler().fit_transform(bc_data)
bc_target = (breast_cancer['diagnosis'].values == 'M').astype(np.uint8)
print(bc_data.shape)
print(bc_target.shape)


# In[21]:


plt.style.use('dark_background')
breast_cancer.hist(figsize=(20,20))
plt.show()


# In[53]:


test_split = 0.2
cutoff = int(bc_data_scaled.shape[0] * (1-test_split))
print('Training on {0} samples, Testing on {1}'.format(cutoff, bc_data_scaled.shape[0]-cutoff))
X_train, X_test = bc_data_scaled[:cutoff], bc_data_scaled[cutoff:]
y_train, y_test = bc_target[:cutoff], bc_target[cutoff:]


# ## Scikit-Learn implementation

# In[56]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
acc = logreg.score(X_test, y_test)
print('test acc {}'.format(acc))
conf_mx = confusion_matrix(y_test, logreg.predict(X_test))
display(pd.DataFrame(conf_mx))


# ## TensorFlow implementation

# In[168]:


n_samples, n_features = bc_data.shape
# Data + bias column
bc_data_scaled_b = np.concatenate((np.ones((n_samples,1)),
                                  bc_data_scaled), axis=1)

X = tf.placeholder(tf.float32, shape=(None, n_features+1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
w = tf.Variable(tf.random_uniform([n_features+1, 1], -1.0,1.0), name='w')
logits = tf.sigmoid(tf.matmul(X, w))
y_pred = tf.cast(tf.greater(logits, 0.5), dtype=tf.float32)
# Cross Entropy Loss
# add 1e-10 to logits to prevent taking a log(0.0) == NaN
x_ent = -1.0/n_samples * (tf.matmul(tf.transpose(y), tf.log(logits + 1e-10)) + tf.matmul(tf.transpose(1.0-y), tf.log(1.0-logits + 1e-10)))
acc = tf.reduce_sum(tf.cast(tf.equal(y_pred, y), dtype=tf.float32)) / float(n_samples)


# In[169]:


opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_op = opt.minimize(x_ent)


# In[191]:


init = tf.global_variables_initializer()
train_loss = []
train_acc = []
n_ep = 500

with tf.Session() as sess:
    sess.run(init)
        
    fd = {X: bc_data_scaled_b, 
          y: np.expand_dims(bc_target, axis=1)}
    
    for ep in range(n_ep):        
        sess.run(train_op, feed_dict=fd)
        ep_loss = x_ent.eval(feed_dict=fd)[0][0]
        ep_acc = acc.eval(feed_dict=fd)
        
        if ep % 50 == 0:
            print('{0} x_ent = {1}; acc = {2}'.format(ep, ep_loss, ep_acc))
        
        train_loss.append(ep_loss)
        train_acc.append(ep_acc)

    pred_logits = logits.eval(feed_dict=fd)
    pred = pred_logits > 0.5
    print(accuracy_score(bc_target, pred))
    display(pd.DataFrame(confusion_matrix(bc_target, pred)))


# In[193]:


plt.style.use('dark_background')
plt.plot(train_acc, 'c')
plt.plot(np.array(train_loss).reshape(-1), 'r')
plt.plot([0,n_ep], [1.0,1.0], 'c--')
plt.xlabel('Training Epoch')
plt.title('Log Reg tf')
plt.legend(['train acc', 'train loss'])
plt.show()

# In[ ]:





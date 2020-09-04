#!/usr/bin/env python
# coding: utf-8

# ## Importing Important Libraries

# In[1]:


import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from IPython.display import HTML


# In[2]:


get_ipython().run_line_magic('cd', '"C:\\\\Users\\\\Admin')


# In[3]:


pwd #File path


# In[4]:


data = pd.read_table('loRe.csv') #reading file using pandas


# In[5]:


features = data.iloc[:,:6]
features #display features


# In[6]:


target = data.iloc[:,-1]    #drag coefficient target values
pd.DataFrame(target)


# In[7]:


df = pd.concat([features,target],axis=1)  #concatinating the feature and target values
df


# In[8]:


df.describe().round(decimals=4) #describing data 


# In[9]:


#Using Pearson Correlation
import seaborn as sns
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[10]:


#Correlation with output variable
cor_target = abs(cor["Cdv"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features


# In[11]:


corr = df.corr('pearson')   #to get the corelation between features and target variable
corrs = [abs(corr[attr]['Cdv']) for attr in list(features)]
l = list(zip(corrs,list(features)))
l.sort(key = lambda x : x[0], reverse=True)
corrs, labels = list(zip((*l)))
index = np.arange(len(labels))
plt.figure(figsize = (15,5))
plt.bar(index, corrs, width =0.5)
plt.xlabel('Attributes')
plt.ylabel('Corelation with target variable')
plt.xticks(index,labels)
plt.show()


# In[12]:


X = df['r1'].values #r1 has highest corelation with Cdv
Y = df['Cdv'].values


# In[13]:


print(Y[:5]) #printing values


# In[14]:


x_scaler = MinMaxScaler() #scaling the target values
X = x_scaler.fit_transform(X.reshape(-1,1))
X = X[:,-1]
y_scaler = MinMaxScaler()
Y = y_scaler.fit_transform(Y.reshape(-1,1))
Y = Y[:,-1]


# In[15]:


print(Y[:5])


# ## Split the data

# In[16]:


xtrain,xtest,ytrain,ytest = train_test_split(X, Y, test_size =0.2) #splitting data into train and test set


# In[17]:


def error(m,x,c,t):   
    N = x.size
    e = sum(((m*x + c) - t)**2)
    return e*1/(2*N)  #evaluating error 


# In[18]:


def update(m,x,c,t,learning_rate):
    grad_m = sum(2*((m*x + c)-t)*x)
    grad_c = sum(2*((m*x + c)-t))
    m = m - grad_m*learning_rate
    c = c -grad_c*learning_rate
    return m,c #updating hyperparameter


# In[19]:


def gradient_descent(init_m,init_c,x,t,learning_rate,iterations,error_threshold):
    m = init_m
    c = init_c
    error_values = list()
    mc_values = list()
    for i in range(iterations):
        e = error(m,x,c,t)
        if e<error_threshold:
            print('Error less then gradient descent, Stopping gradient descent.')
            break
        error_values.append(e)
        m,c = update(m,x,c,t,learning_rate)
        mc_values.append((m,c))
    return m,c,error_values, mc_values #learning algorithm


# In[20]:


get_ipython().run_cell_magic('time', '', 'init_m = 0\ninit_c = 0\nlearning_rate = 0.001\niterations = 250\nerror_threshold = 0.001\nm,c,error_values,mc_values = gradient_descent(init_m,init_c,xtrain,ytrain,learning_rate,iterations,error_threshold)')


# ## Visualization of trainning model

# In[21]:


mc_values_anim = mc_values[0:250:5]


# In[22]:


fig, ax = plt.subplots()
ln, = plt.plot([],[],'ro-', animated = True)

def init():
    plt.scatter(xtest,ytest,color = 'g')
    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.0)
    return ln,

def update_frame(frame):
    m,c = mc_values_anim[frame]
    x1,y1 = -0.5, m*-.5 + c
    x2,y2 = 1.5, m*1.5 + c
    ln.set_data([x1,x2],[y1,y2])
    return ln,

anim = FuncAnimation(fig, update_frame, frames = range(len(mc_values_anim)), init_func = init, blit = True)

HTML(anim.to_html5_video())


# In[23]:


plt.scatter(xtrain,ytrain,color='b')
plt.plot(xtrain,(m*xtrain + c), color='r')


# In[24]:


plt.plot(np.arange(len(mc_values)),error_values)
plt.xlabel('Iterations')
plt.ylabel('Error')


# ## Prediction

# In[25]:


predicted = (m*xtest) + c


# In[26]:


mean_squared_error(ytest, predicted)


# In[27]:


p = pd.DataFrame(list(zip(xtest,ytest,predicted)),columns = ['x','target_y','predicted_y'])
p.head()


# In[28]:


plt.scatter(xtest, ytest, color ='b')
plt.plot(xtest, predicted, color ='r')


# ## Prediction of Drag Coefficient

# In[29]:


predicted = predicted.reshape(-1,1)
xtest = xtest.reshape(-1,1)
ytest = ytest.reshape(-1,1)

xtest_scaled = x_scaler.inverse_transform(xtest)
ytest_scaled = y_scaler.inverse_transform(ytest)
predicted_scaled = y_scaler.inverse_transform(predicted)

xtest_scaled = xtest_scaled[:,-1]
ytest_scaled = ytest_scaled[:,-1]
predicted_scaled = predicted_scaled[:,-1]

p = pd.DataFrame(list(zip(xtest_scaled,ytest_scaled,predicted_scaled)),columns = ["x","target_y","predicted_y"])
p = p.round(decimals =2)
p.head()


# ### From above table we can say that we can succesfully predict the Volumetric coefficient of drag for given reynolds number for different features for random envolope. (Here we have taken length of envolope as constant)

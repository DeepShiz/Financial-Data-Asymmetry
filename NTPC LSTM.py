#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[4]:


dataset_train = pd.read_csv('train.csv')
dataset_train.head()


# In[5]:


training_set = dataset_train.iloc[:, 5:6].values
print(training_set)


# In[6]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


# In[7]:


X_train = []
y_train = []
for i in range(60, 2000):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[8]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[9]:


regressor = Sequential()

regressor.add(LSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 60))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 1000, batch_size = 32)


# In[10]:


dataset_test = pd.read_csv('test2.csv')
real_stock_price = dataset_test.iloc[:, 5:6].values


# In[48]:


dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis =0)
print(dataset_total)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
#print(inputs)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 760) :
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# In[41]:


plt.plot(real_stock_price, color = 'black', label = 'NTPC Price')
plt.plot(predicted_stock_price, color = 'red', label = 'Predicted NTPC Price')
plt.title('NTPC  Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('NTPC Stock Price')
plt.legend()
plt.show()


# In[30]:


dataset_test1 = pd.read_csv('test1.csv')


# In[14]:


inputt=[]
inputt=dataset_test1['Close'].values
inputt = inputt.reshape(-1,1)


# In[15]:


inputt.shape


# In[16]:


inputt = sc.transform(inputt)
X_test1 = []
for i in range(60, 61) :
    X_test1.append(inputt[i-60:i, 0])
X_test1 = np.array(X_test1)
X_test1 = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[1], 1))
predicted_stock_price1 = regressor.predict(X_test1)
predicted_stock_price1 = sc.inverse_transform(predicted_stock_price1)


# In[17]:


print(predicted_stock_price1)






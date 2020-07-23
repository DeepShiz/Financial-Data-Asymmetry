# Financial-Data-Asymmetry
Architecture of Financial data asymmetry: 1.Many-to-One Sequence Problems with a Single Feature: Here I have LSTM model for time series data forcasting and we have Data of a company with the stock prices of past 'Y' days closing,opening,high,low. Firstly we have to make our data perfect,for that extract only the closing stock price coloumn and store in an array. 
  Preparing our data for the model:
~Split the data into train and test,let us take there are 'X' days of closing stock price data.create data in 'N'time steps and convert into an array.Then as the input should be in 3D shape ,we reshape our data into [samples, time-steps, features] if having 1600 days of data which means 'X' samples and usually the time steps are taken in the range of 50-80. The feature used is only one which is taken into consideration is 'closing time'. Now this will create a 3D form to our input which is accepted by lstm 
Building LSTM
~After cleaning the data and making it ready, build a LSTM model, first importing modules from 'keras' lib like sequential(for initializing the network),dense(adding densely connected network layer),lstm and dropout(this adds dropout layer which prevents over fitting)
~Now initializing Lstm layer with:(no. of neurons ,return sequence and input shape)
 *No. of neurons are usually equal or less than no. of samples taken at from the data to train.we can increase the no. of hidden layers to and and assign less no of neuron at a time for one layer, but by systematically assigning the neurons in a layer n to the next layer this thing can be solved eg:  (if we have 2000 data points then assign first layer with 100 neuron and the next with 20)
~After initializing connect a dropout layer with specifiyng small percentage to dropout(I found out this layer can be skiped as it can reduce the performance of the model) 
~Then connect hidden dense layer with activation function 'ReLu' which will work for our model and also with the no. of neurons in this layer as explained earlier in the initialization
~final layer dense layer (output layer) is made with same 'ReLu' activation function and here the No. of neuron we use is '1' as we need only one output (next day stock price).
~NOW for compilation use 'ADAM' optimizer, here the loss is 'mean sq error' ,adam optimizer works on each parameter to give more accurate mean sq error which is enough for our model.
~FIT the model with input data created and let the no. of epochs and batch size,
 *Making prediction test data
~Making same changes as made in train data ,creating same no. of time steps and converting the data in 3D shape etc.
 *Predicting 
~predict the stock price from the test data
* Check 
comparing the actual price to the predicted price would suggest the capability off  this LSTM  architecture

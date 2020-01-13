import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense

training_data = pd.read_csv('Training_Data_Labels.csv')

train_y = training_data.iloc[:,11:12].values
train_x = training_data.drop(columns = ['label']).values

ohe = OneHotEncoder()
train_y = ohe.fit_transform(train_y)

sc = StandardScaler()
train_x = sc.fit_transform(train_x)

train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size = 0.1)

# Neural network
model = Sequential()
model.add(Dense(12, input_dim=14, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# training
history = model.fit(train_x, train_y, epochs=100, batch_size=64)

# test
pred_test = model.predict(test_x)

# Results
#Converting predictions to label
pred = list()
for i in range(len(pred_test)):
    pred.append(np.argmax(pred_test[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(test_y.shape[0]):
    test.append(np.argmax(test_y[i]))

# Accuracy
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)
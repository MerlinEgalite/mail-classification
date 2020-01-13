import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense

training_data = pd.read_csv('datasets/new_train.csv')

train_y = training_data.iloc[:,11:12].values
train_x = training_data.drop(columns = ['Id','label']).values

ohe = OneHotEncoder()
train_y = ohe.fit_transform(train_y)

sc = StandardScaler()
train_x = sc.fit_transform(train_x)

train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size = 0.1)

# Neural network
model = Sequential()
model.add(Dense(12, input_dim=13, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# training
history = model.fit(train_x, train_y, epochs=400, batch_size=64)

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

# Official Kaggle test
test_data = pd.read_csv('datasets/new_test.csv')
test_kaggle = test_data.drop(columns = ['Id']).values
test_kaggle = sc.fit_transform(test_kaggle)

pred_kaggle = model.predict(test_kaggle)

# Results
#Converting predictions to label
pred = list()
for i in range(len(pred_kaggle)):
    pred.append(np.argmax(pred_kaggle[i]))

pred_to_submit = pd.DataFrame(pred, columns=['label'])
pred_to_submit.to_csv("datasets/neural_networks_submission.csv", index=True, index_label='Id')

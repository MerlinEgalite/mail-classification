"""
This script can be used as skeleton code to read the challenge train and test
"""
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

## Read csvs
train_df = pd.read_csv('train.csv', index_col=0)
test_df = pd.read_csv('test.csv', index_col=0)

## Handle missing values
train_df.fillna('NA', inplace=True)
test_df.fillna('NA', inplace=True)

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

new_train_df = train_df.copy()
new_train_df = new_train_df.drop(['date', 'bcced'], axis=1)

new_test_df = test_df.copy()
new_test_df = new_test_df.drop(['date', 'bcced'], axis=1)


## Function for processing data
def processing(dataset, new_dataset):

    for (index, row) in dataset['date'].items():

        # Process for date
        date_list = row.split(' ')
        if len(date_list) > 4:
            day = date_list[0][0:3]
            month = date_list[2]
            hour = float(date_list[4][0:2])
            minutes = float(date_list[4][3:5]) / 60
            new_hour = hour + minutes
            new_dataset.at[index, 'day'] = day
            new_dataset.at[index, 'month'] = month
            new_dataset.at[index, 'hour'] = new_hour

        # Process for mail_type
        new_dataset.at[index, 'mail_type'] = new_dataset.at[index, 'mail_type'].lower()
        new_dataset.at[index, 'org'] = new_dataset.at[index, 'org'].lower()
        new_dataset.at[index, 'tld'] = new_dataset.at[index, 'tld'].lower()

    return new_dataset


## Modify the Train & Test sets
new_train_df = processing(train_df, new_train_df)
new_test_df = processing(test_df, new_test_df)


## Save into csv files
new_train_df.to_csv("new_train.csv", index=True, index_label='Id')
new_test_df.to_csv("new_test.csv", index=True, index_label='Id')

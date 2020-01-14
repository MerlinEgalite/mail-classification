import pandas as pd
import re
import numpy as np
import missingno as msno

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

months = {'JAN':'Jan', 'FEV':'Feb', 'MAR':'Mar', 'AVR':'Apr', 'MAI':'May', 'JUN':'Jun', 'JUI':'Jul', 'AOU':'Aug', 'SEP':'Sep', 'OCT':'Oct', 'NOV':'Nov', 'DEC':'Dec'}

# Read csv
train_df = pd.read_csv('datasets/train.csv', index_col=0)
test_df = pd.read_csv('datasets/test.csv', index_col=0)

# Handle missing values
train_df.fillna('NA', inplace=True)
test_df.fillna('NA', inplace=True)

new_train_df = train_df.copy()
new_train_df = new_train_df.drop(['date', 'bcced'], axis=1)

new_test_df = test_df.copy()
new_test_df = new_test_df.drop(['date', 'bcced'], axis=1)

# Label Encoder
def LabelEnc(data_df, title_column):
    """Return a new pandas dataframe with label encoder"""
    lab_enc = LabelEncoder()
    new_data_df = data_df.copy()
    new = lab_enc.fit_transform(new_data_df[title_column].astype(str))
    new_data_df[title_column] = new
    return new_data_df[title_column]


# Function for processing data
def processing(dataset, new_dataset):

    for (index, row) in dataset['date'].items():

        date_list = row.split(' ')

        # Prevent empty string
        if '' in date_list:
            date_list.remove('')

        # Process for date column
        # Regex expression to match the right date expression
        prog = re.compile('''(\")?([a-zA-Z]{3}), +([0-9]{1,2}) +([a-zA-Z]{3}) +([0-9]{4}) +([0-9:]{8})''')
        result = prog.match(row)

        if result is not None:
            # Get the day, date, month and create new columns
            day = date_list[0][0:3]
            month = date_list[2]
            hour = float(date_list[4][0:2])
            minutes = float(date_list[4][3:5]) / 60
            new_hour = hour + minutes
            new_dataset.at[index, 'day'] = day
            new_dataset.at[index, 'month'] = month
            new_dataset.at[index, 'hour'] = new_hour

        else:

            prog = re.compile('''([0-9]{1,2}) +([a-zA-Z]{3}) +([0-9]{4}) +([0-9:]{8})''')
            result = prog.match(row)

            if result is not None:

                hour = float(date_list[3][0:2])
                minutes = float(date_list[3][3:5]) / 60
                new_hour = hour + minutes
                new_dataset.at[index, 'day'] = ""
                new_dataset.at[index, 'month'] = month
                new_dataset.at[index, 'hour'] = new_hour

            else:

                month = months[date_list[0][3:6]]
                hour = float(date_list[1][0:2])
                minutes = float(date_list[1][3:5]) / 60
                new_hour = hour + minutes
                new_dataset.at[index, 'day'] = ""
                new_dataset.at[index, 'month'] = month
                new_dataset.at[index, 'hour'] = new_hour


        # Process for mail_type
        new_dataset.at[index, 'mail_type'] = new_dataset.at[index, 'mail_type'].lower()
        new_dataset.at[index, 'org'] = new_dataset.at[index, 'org'].lower()
        new_dataset.at[index, 'tld'] = new_dataset.at[index, 'tld'].lower()

    # Label Encoder to transform categories into numbers
    categorical = ['day', 'month', 'org', 'tld', 'mail_type']
    for cat in categorical:
        new_dataset[cat] = LabelEnc(new_dataset, cat)

    return new_dataset


# Modify the Train & Test sets
new_train_df = processing(train_df, new_train_df)
new_test_df = processing(test_df, new_test_df)


imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
new_train_df[['org', 'hour', 'day', 'month', 'salutations']] = imputer.fit_transform(new_train_df[['org', 'hour', 'day', 'month', 'salutations']])


msno.matrix(new_train_df)

# Save into csv files
new_train_df.to_csv("datasets/new_train.csv", index=True, index_label='Id')
new_test_df.to_csv("datasets/new_test.csv", index=True, index_label='Id')

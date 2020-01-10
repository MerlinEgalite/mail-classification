"""
This script can be used as skeleton code to read the challenge train and test
"""
import pandas as pd
import re

from sklearn.preprocessing import LabelEncoder

# Read csv
train_df = pd.read_csv('train.csv', index_col=0)
test_df = pd.read_csv('test.csv', index_col=0)

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

        # Process for date column
        # Regex expression to match the right date expression
        prog = re.compile('''(\")?([a-zA-Z]{3}), +([0-9]{1,2}) ([a-zA-Z]{3}) ([0-9]{4}) ([0-9:]{8})''')
        result = prog.match(row)
        if result is not None:
            # Get the day, date, month and create new columns
            date_list = row.split(' ')
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

    # Label Encoder to transform categories into numbers
    categorical = list(new_dataset)
    for cat in categorical:
        new_dataset[cat] = LabelEnc(new_dataset, cat)

    return new_dataset


# Modify the Train & Test sets
new_train_df = processing(train_df, new_train_df)
new_test_df = processing(test_df, new_test_df)

# Save into csv files
new_train_df.to_csv("new_train.csv", index=True, index_label='Id')
new_test_df.to_csv("new_test.csv", index=True, index_label='Id')

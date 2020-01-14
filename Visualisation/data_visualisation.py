"""
This script can be used as skeleton code to read the challenge train and test
csvs, to train a trivial model, and write data to the submission file.
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

## Filtering column "mail_type"
train_x = train_df[['mail_type', 'org', 'tld']]
train_y = train_df[['label']]

#bcced = train_df[['bcced']]
#it seems that bcced is not useful to add

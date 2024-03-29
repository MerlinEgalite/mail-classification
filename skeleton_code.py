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
train_x = train_df[['mail_type']]
train_y = train_df[['label']]

test_x = test_df[['mail_type']]

## Do one hot encoding of categorical feature
feat_enc = OneHotEncoder(handle_unknown='ignore')
feat_enc.fit(train_x)
train_x_featurized = feat_enc.transform(train_x)
test_x_featurized = feat_enc.transform(test_x)

## Train a simple KNN classifier using featurized data
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_x_featurized, train_y)
pred_y = neigh.predict(test_x_featurized)

## Save results to submission file
pred_df = pd.DataFrame(pred_y, columns=['label'])
pred_df.to_csv("knn_sample_submission.csv", index=True, index_label='Id')

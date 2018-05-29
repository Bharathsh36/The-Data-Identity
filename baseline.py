import pandas as pd
import numpy as np

import lightgbm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

train = pd.read_csv('../feature/train_feature.csv')
test = pd.read_csv('../feature/test_feature.csv')

print(train.shape,test.shape)

y = train.is_pass.values
train.drop(['id', 'is_pass'], inplace=True, axis=1)

x, x_test, y, y_test = train_test_split(train, y, test_size=0.2, random_state=42, stratify=y)

train_data = lightgbm.Dataset(x, label=y)
test_data = lightgbm.Dataset(x_test, label=y_test)

parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'verbose': 0
}

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=1000,
                       early_stopping_rounds=50)

ids = test['id'].values
test.drop('id', inplace=True, axis=1)

x = test.values
y = model.predict(x)

output = pd.DataFrame({'id': ids, 'is_pass': y})
output.to_csv("../output/submission.csv", index=False)

gain = model.feature_importance('gain')
ft = pd.DataFrame({'feature':model.feature_name(), 'split':model.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(ft.head(25))

plt.figure()
ft[['feature','gain']].head(15).plot(kind='barh', x='feature', y='gain', legend=False, figsize=(10, 20))
plt.gcf().savefig('features_importance.png')
# -*- coding:utf-8 -*-
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer

train_data = pd.read_csv('alldata.csv')
test_data = pd.read_csv('testdata.csv')


# 将数据转化未label（0-N）形式
def encode_features(df_train, df_test):
    features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    df_combined = pd.concat([df_train[features], df_test[features]])
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test


def simplify_interval_info(df):
    bins_density = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    bins_sugar = (0, 0.1, 0.2, 0.3, 0.4, 0.5)

    group_name_density = [0, 1, 2, 3, 4, 5, 6, 7]
    group_name_sugar = [0, 1, 2, 3, 4]

    category_density = pd.cut(df['密度'], bins_density, labels=group_name_density)
    categroy_sugar = pd.cut(df['含糖率'], bins_sugar, labels=group_name_sugar)

    df['密度'] = category_density
    df['含糖率'] = categroy_sugar

    return df


train_data, test_data = encode_features(train_data, test_data)
train_data = simplify_interval_info(train_data)
test_data = simplify_interval_info(test_data)

X_all = train_data.drop(['好瓜'], axis=1)
y_all = train_data['好瓜']
y_result = [1, 0]

num_test = 0.35
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=3)
# Choose some parameter combinations to try
parameters = {'n_estimators': [5, 6, 7],
              'criterion': ['entropy', 'gini']
              }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)
clf = RandomForestClassifier()

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_
print(X_train)
clf = clf.fit(X_train, y_train)
test_predictions = clf.predict(X_test)
print("测试集准确率:  %s " % accuracy_score(y_test, test_predictions))
# 以下为实际测试准确率
test_data_result = test_data.drop(['好瓜'], axis=1)
y_result = test_data['好瓜']
predictions = clf.predict(test_data_result)
print("最终准确率:  %s " % accuracy_score(y_result, predictions))

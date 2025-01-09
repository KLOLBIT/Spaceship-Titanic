import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test['PassengerId']

sns.heatmap(data.isnull(), cbar = False, yticklabels = False)
#plt.show()

data.drop('Name', axis = 1, inplace = True)
test.drop('Name', axis = 1, inplace = True)
# print(data.info())

data['Groups'] = data['PassengerId'].str[:4]
test['Groups'] = data['PassengerId'].str[:4]
data['The_Numbers'] = data['PassengerId'].str[5:]
test['The_Numbers'] = data['PassengerId'].str[5:]

def cleaning_service(data):
    cols = ['CryoSleep', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    cat_cols = ['HomePlanet', 'Cabin', 'Destination', 'Groups', 'The_Numbers']
    unique_destinations = data['Destination'].unique()
    unique_vip = data['VIP'].unique()
    unique_groups = data['Groups'].unique()
    unique_numbers = data['The_Numbers'].unique()
    unique_cryosleep = data['CryoSleep'].unique()
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)
    for i in cat_cols:
        data[i] = data[i].fillna(data[i].mode()[0])
    data['Deck'] = data['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
    data['Port'] = data['Cabin'].apply(lambda s: s[-1] if pd.notnull(s) else 'M')
    data["Deck"] = data["Deck"].map({'B':0, 'F':1, 'A':2, 'G':3, 'E':4, 'D':5, 'C':6, 'T':7}).astype(int)
    data["Port"] = data["Port"].map({'P':0, 'S':1}).astype(int)
    data.drop(['Cabin'], axis=1, inplace=True)
    data['Groups'] = data['Groups'].map(dict(zip(unique_groups, list(range(len(unique_groups)))))).astype(int)
    data['The_Numbers'] = data['The_Numbers'].map(dict(zip(unique_numbers, list(range(len(unique_numbers)))))).astype(int)
    data["HomePlanet"] = data["HomePlanet"].map({'Earth':0, 'Europa':1, 'Mars':2}).astype(int)
    data["Destination"] = data["Destination"].map(dict(zip(unique_destinations,list(range(len(unique_destinations)))))).astype(int)
    data["VIP"] = data["VIP"].map(dict(zip(unique_vip,list(range(len(unique_vip)))))).astype(int)
    data["CryoSleep"] = data["CryoSleep"].map(dict(zip(unique_cryosleep,list(range(len(unique_cryosleep)))))).astype(int)
    data.drop('PassengerId', axis=1, inplace=True)
    return data
data = cleaning_service(data)
test = cleaning_service(test)
sns.heatmap(data.isnull(), yticklabels=False, cbar=False)
#plt.show()
#print(data.head())

X = data.drop('Transported', axis = 1)
y = data['Transported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# np.random.seed(42)

# data_shuffled = data.sample(frac=1)
# X1 = data_shuffled.drop('Transported', axis = 1)
# y1 = data_shuffled['Transported']
# # data split by hand in order to increase the score of the model
# train_split = round(0.7 * len(data_shuffled)) #70%
# valid_split = round(train_split + 0.15 * len(data_shuffled)) #15%
# X_train1, y_train1 = X1[:train_split], y1[:train_split]
# X_valid, y_valid = X1[train_split:valid_split], y1[train_split:valid_split]
# X_test1, y_test1 = X1[valid_split:], y1[valid_split:]


balance = np.sum(y) / len(y) # checking the balance of the positive and negative data
#print(balance)

#baseline model
svc = SVC()
XGboost = XGBClassifier()
#print(svc.get_params())

# SETUP RandomizedSearchCV and GridSearchCV
grid = {
    'C': [800],
    'kernel': ['rbf'],
    'class_weight': [None, 'balanced'],
    'gamma': ['scale', 'auto']
}

gridXG = {
    'booster': ['gbtree', 'dart'],
    'eta': [0.3, 0.4, 0.5, 0.8],
    'gamma': [0, 10, 30, 50, 80, 120],
    'max_depth': [6, 12, 20, 32, 64]
}

def finish_line(data, param):
    search = GridSearchCV(
        estimator=data,
        param_grid=param,
        verbose=2,
        n_jobs=-1
        ).fit(X_train, y_train)
    prediction = search.predict(X_test)
    print('accuracy_score: ' + str(accuracy_score(prediction, y_test)))
    print('precision_score: ' + str(precision_score(prediction, y_test)))
    print('f1_score: ' + str(f1_score(prediction, y_test)))
    print(classification_report(prediction, y_test))
    return search

def finish_lineXG(data, param):
    search = GridSearchCV(
        estimator=data,
        param_grid=param,
        verbose=2,
        n_jobs=-1
        ).fit(X_train, y_train)
    prediction = search.predict(X_test)
    print('accuracy_score: ' + str(accuracy_score(prediction, y_test)))
    print('precision_score: ' + str(precision_score(prediction, y_test)))
    print('f1_score: ' + str(f1_score(prediction, y_test)))
    print(classification_report(prediction, y_test))
    return search

model = finish_line(svc, grid)

modelXG = finish_lineXG(XGboost, gridXG)

# svc_rs = RandomizedSearchCV(
#     estimator=svc,
#     param_distributions=grid,
#     n_iter=24,
#     verbose=2,
#     n_jobs=-1
# )
f_pred = model.predict(test)

f_predXG = modelXG.predict(test)

# print('------')

# gnb = GaussianNB().fit(X_train, y_train)
# gnb_prediction = gnb.predict(X_test)
# print('accuracy_score: ' + str(accuracy_score(gnb_prediction, y_test)))
# print('precision_score: ' + str(precision_score(gnb_prediction, y_test)))
# print('f1_score: ' + str(f1_score(gnb_prediction, y_test)))
# f_gnb_pred = gnb.predict(test)

# print('------')

# sgd = SGDClassifier().fit(X_train, y_train)
# sgd_prediction = sgd.predict(X_test)
# print('accuracy_score: ' + str(accuracy_score(sgd_prediction, y_test)))
# print('precision_score: ' + str(precision_score(sgd_prediction, y_test)))
# print('f1_score: ' + str(f1_score(sgd_prediction, y_test)))
# f_sgd_pred = sgd.predict(test)
#kc = KNeighborsClassifier(n_neighbors=18).fit(X_train, y_train)
#kc_prediction = kc.predict(X_test)
#print(accuracy_score(kc_prediction, y_test))
#print(classification_report(kc_prediction, y_test))
#kc_final_pred = kc.predict(test)

# error_rate = []
# for i in range(1, 120):
#     knc = KNeighborsClassifier(n_neighbors=i)
#     knc.fit(X_train, y_train)
#     pred_i = knc.predict(X_test)
#     error_rate.append(np.mean(pred_i != y_test))

# plt.figure(figsize=(10,6))
# plt.plot(range(1, 120), error_rate, color = 'green', linestyle = 'dashed', marker='o', markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# plt.show()

df_XG = pd.DataFrame({
    'PassengerId':test_ids.values,
    'Transported':f_predXG
})

df_svc = pd.DataFrame({
    'PassengerId':test_ids.values,
    'Transported':f_pred
})

#df_kc = pd.DataFrame({
#    'PassengerId':test_ids.values,
#    'Transported':kc_final_pred
#})
df_svc.to_csv('submission_SVC.csv', index=False)

df_XG.to_csv('submission_XGboost.csv', index=False)
# df_kc.to_csv('submission_kc.csv', index = False)

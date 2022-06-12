import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Read data & Split it
df = pd.read_csv("data/train.csv")
#print(pd.DataFrame(df).head(3)) #The first three data
#print(df.columns) #data Index
'''
data Index
['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'],
      dtype='object')
'''
X_df = pd.DataFrame(df, columns = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered'])
y_df = pd.DataFrame(df, columns = ['count'])
X_train, X_valid, y_train, y_valid = train_test_split(X_df, y_df, train_size=0.6, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, train_size=0.5, random_state=0)
#print(X_train, X_valid, X_test, y_train, y_valid, y_test)

#Standardization
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_valid_std = sc.transform(X_valid)
X_test_std = sc.transform(X_test)

#XGBoost
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
n_estimators = [10,20,30,40,50,60,70,80,90,100]
max_depth = [1,2,3,4,5,6]
parameters_to_search = {'n_estimators': n_estimators, 
              'max_depth': max_depth} #設定要訓練的值
xgbModel = xgb.XGBRegressor(n_estimators = 100, max_depth = 6)
xgbModel_cv = GridSearchCV(xgbModel, parameters_to_search, cv=5) #可以直接找出最佳的訓練值
xgbModel_cv.fit(X_train_std, y_train.values.ravel())
xgbScore = xgbModel_cv.score(X_test_std, y_test.values.ravel())
print('Correct rate using XGBoost: {:.5f}'.format(xgbScore))

#Use MSE sure whether it has overfitting.
print("XGBoost's MSE")
from sklearn import metrics
train_pred = xgbModel_cv.predict(X_train_std)
xgbTrainMse = metrics.mean_squared_error(y_train.values.ravel(), train_pred)
print('train data MSE: ', xgbTrainMse)
test_pred = xgbModel_cv.predict(X_test_std)
xgbTestMse = metrics.mean_squared_error(y_test.values.ravel(), test_pred)
print('test data MSE: ', xgbTestMse)

#SVR
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
#import numpy as np
svr_maxRate = 0 #驗證及訓練結果的最高正確率
svr_c = 2 #最適合用在此的SVR參數
svr_epsilon = 0.5 #最適合用在此的SVR參數
#rng = np.random.RandomState(0)
svrModel = make_pipeline(StandardScaler(), SVR(C=svr_c, epsilon=svr_epsilon))
svrModel.fit(X_train_std, y_train.values.ravel())
for i in range(1, 5):
    for j in range(0, 5):
        svrModel = make_pipeline(StandardScaler(), SVR(C=i, epsilon = j/10))
        svrModel.fit(X_train_std, y_train.values.ravel())
        if svr_maxRate < svrModel.score(X_valid_std, y_valid.values.ravel()):
            svr_maxRate = svrModel.score(X_valid_std, y_valid.values.ravel())
            svr_c = i
            svr_epsilon = j/10
svrModel = make_pipeline(StandardScaler(), SVR(C = svr_c, epsilon = svr_epsilon))
svrModel.fit(X_train_std, y_train.values.ravel())
svrScore = svrModel.score(X_test_std, y_test.values.ravel())
print('Correct rate using SVR: {:.5f}'.format(svrScore))

#Use MSE sure whether it has overfitting.
print("SVR's MSE")
from sklearn import metrics
train_pred = svrModel.predict(X_train_std)
svrTrainMse = metrics.mean_squared_error(y_train.values.ravel(), train_pred)
print('train data MSE: ', svrTrainMse)
test_pred = svrModel.predict(X_test_std)
svrTestMse = metrics.mean_squared_error(y_test.values.ravel(), test_pred)
print('test data MSE: ', svrTestMse)

# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_maxRate = 0
rf_state = 0
for i in range (1,10):
    rfModel = RandomForestRegressor(random_state = i)
    rfModel.fit(X_train_std, y_train.values.ravel())
    if rf_maxRate < rfModel.score(X_valid_std, y_valid.values.ravel()):
        rf_maxRate = rfModel.score(X_valid_std, y_valid.values.ravel())
        rf_state = i
RandomForestRegressor(random_state = rf_state)
rfScore = round(rfModel.score(X_test_std, y_test.values.ravel()),5)
print("Correct rate using Random Forest: ", rfScore)

#Use MSE sure whether it has overfitting.
print("Random Forest's MSE")
from sklearn import metrics
train_pred = rfModel.predict(X_train_std)
rfTrainMse = metrics.mean_squared_error(y_train.values.ravel(), train_pred)
print('train data MSE: ', rfTrainMse)
test_pred = rfModel.predict(X_test_std)
rfTestMse = metrics.mean_squared_error(y_test.values.ravel(), test_pred)
print('test data MSE: ', rfTestMse)

#Stacking
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
#弱學習器
estimators = [
    ('xgb', GridSearchCV(xgbModel, parameters_to_search, cv=5)),
    ('svr', make_pipeline(StandardScaler(), SVR(C = svr_c, epsilon = svr_epsilon))),
    ('rf', RandomForestRegressor(random_state = rf_state))
]
#Stacking將不同模型優缺點進行加權，讓模型更好。
#final_estimator：集合所有弱學習器訓練出最終預測模型。預設為LogisticRegression。
stackModel = StackingRegressor(
    estimators=estimators, final_estimator= MLPRegressor(activation = "relu", alpha = 0.1, hidden_layer_sizes = (8,8),
                            learning_rate = "constant", max_iter = 200, random_state = 100)
)
stackModel.fit(X_train_std, y_train.values.ravel())

#Use MSE sure whether it has overfitting.
print("After stacking, it's MSE")
from sklearn import metrics
train_pred = stackModel.predict(X_train_std)
stackTrainMse = metrics.mean_squared_error(y_train.values.ravel(), train_pred)
print('train data MSE: ', stackTrainMse)
test_pred = stackModel.predict(X_test_std)
stackTestMse = metrics.mean_squared_error(y_test.values.ravel(), test_pred)
print('test data MSE: ', stackTestMse)

stackScore = stackModel.score(X_test_std, y_test.values.ravel())
print("Correct rate after Stacking: ", stackScore)

#Use dataframe output all result
result = {
    "Model": ["score", "train data mse", "test data mse"],
    "XGBoost": [xgbScore, xgbTrainMse, xgbTestMse],
    "SVR": [svrScore, svrTrainMse, svrTestMse],
    "Random Forest": [rfScore, rfTrainMse, rfTestMse],
    "After Stacking":[stackScore, stackTrainMse, stackTestMse]
}
result_df = pd.DataFrame(result)
print(result_df)

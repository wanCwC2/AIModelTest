import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

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

#XGBoost基本設定
n_estimators = [10,20,30,40,50,60,70,80,90,100]
max_depth = [1,2,3,4,5,6]
parameters_to_search = {'n_estimators': n_estimators, 
              'max_depth': max_depth} #設定要訓練的值
xgbModel = xgb.XGBRegressor(n_estimators = 100, max_depth = 6)

#Stacking
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
svr_c = 4
svr_epsilon = 0.0
rf_state = 9
stack_minMse = 9999
stack_iter = 1400
stack_state = 400
#弱學習器
estimators = [
    ('xgb', GridSearchCV(xgbModel, parameters_to_search, cv=5)),
    ('svr', make_pipeline(StandardScaler(), SVR(C = svr_c, epsilon = svr_epsilon))),
    ('rf', RandomForestRegressor(random_state = rf_state))
]
#Stacking將不同模型優缺點進行加權，讓模型更好。
#final_estimator：集合所有弱學習器訓練出最終預測模型。預設為LogisticRegression。

for i in range (200, 2001, 600):
    for j in range (100, 1001 ,300):
        stackModel = StackingRegressor(
            estimators=estimators, final_estimator= MLPRegressor(activation = "relu", alpha = 0.1, hidden_layer_sizes = (8,8),
                                        learning_rate = "constant", max_iter = i, random_state = j)
            )
        stackModel.fit(X_train_std, y_train.values.ravel())
        train_pred = stackModel.predict(X_train_std)
        stackTrainMse = metrics.mean_squared_error(y_train.values.ravel(), train_pred)
        valid_pred = stackModel.predict(X_valid_std)
        stackTestMse = metrics.mean_squared_error(y_valid.values.ravel(), valid_pred)
        if stack_minMse > abs(stackTrainMse - stackTestMse):
            stack_minMse = abs(stackTrainMse - stackTestMse)
            stack_iter = i
            stack_state = j
        print("Done! i = ",i," j = ",j)
        print('train data MSE: ', stackTrainMse)
        print('test data MSE: ', stackTestMse)

stackModel = StackingRegressor(
    estimators=estimators, final_estimator= MLPRegressor(activation = "relu", alpha = 0.1, hidden_layer_sizes = (8,8),
                            learning_rate = "constant", max_iter = stack_iter, random_state = stack_state)
)
stackModel.fit(X_train_std, y_train.values.ravel())
stackScore = stackModel.score(X_test_std, y_test.values.ravel())
print("Correct rate after Stacking: ", stackScore)

#Use MSE sure whether it has overfitting.
print("After stacking, it's MSE")
train_pred = stackModel.predict(X_train_std)
stackTrainMse = metrics.mean_squared_error(y_train.values.ravel(), train_pred)
print('train data MSE: ', stackTrainMse)
test_pred = stackModel.predict(X_test_std)
stackTestMse = metrics.mean_squared_error(y_test.values.ravel(), test_pred)
print('test data MSE: ', stackTestMse)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split

#Read data & Split it
df = pd.read_csv("data/train.csv")
#print(pd.DataFrame(df).head(3)) #The first three data
#print(df.columns) #data Index
'''
data Index
['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'],
      dtype='object')
'''
x_df = pd.DataFrame(df, columns = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered'])
y_df = pd.DataFrame(df, columns = ['count'])
x_train, x_valid, y_train, y_valid = train_test_split(x_df, y_df, train_size=0.6, random_state=0)
x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, train_size=0.5, random_state=0)
#print(x_train, x_valid, x_test, y_train, y_valid, y_test)

##KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
'''
# 設定需要搜尋的K值，'n_neighbors'是sklearn中KNN的引數
parameters={'n_neighbors':for i in range(1,10)}
knn=KNeighborsClassifier()#注意：這裡不用指定引數

# 通過GridSearchCV來搜尋最好的K值。這個模組的內部其實就是對每一個K值進行評估
clf=GridSearchCV(knn,parameters,cv=5)  #5折
clf.fit(x_train,y_train)

# 輸出最好的引數以及對應的準確率
print("最終最佳準確率：%.2f"%clf.best_score_,"最終的最佳K值",clf.best_params_)

kng=KNeighborsRegressor(n_neighbors=1)

kng.fit(x_train,y_train)
prediction=knn.predict(x_test)

kng_test_score=kng.score(x_test,y_test)

print('test data score:{:.2f}'.format(kng_test_score))
'''
error_rate = []
for i in range(1,60):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_valid)
    error_rate.append(np.mean(pred_i != y_valid.T))
    '''
    prediction=knn.predict(x_test)
    knn_test_score=knn.score(x_test,y_test)
    print('test data score:{:.2f}'.format(knn_test_score))
    '''

#將k=1~60的錯誤率製圖畫出
plt.figure(figsize=(10,6))
plt.plot(range(1,60),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

prediction=knn.predict(x_test)
knn_test_score=knn.score(x_test,y_test)
print('test data score:{:.2f}'.format(knn_test_score))
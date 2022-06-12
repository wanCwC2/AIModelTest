import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

#from KNN import KNN

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

#KNN
# Method 1: Use GridSearchCV to find the best value of k
# 設定需要搜尋的K值，'n_neighbors'是sklearn中KNN的引數
parameters={'n_neighbors':[1,3,5,7,9,11,13]}
knn=KNeighborsRegressor() #注意：這裡不用指定引數
# 通過GridSearchCV來搜尋最好的K值。這個模組的內部其實就是對每一個K值進行評估
clf=GridSearchCV(knn,parameters,cv=5) #5折
clf.fit(X_train,y_train)
# 輸出最好的引數以及對應的準確率
#print("The best rate is ：%.5f"%clf.best_score_,"The best value of k is",clf.best_params_)

#Method 2: Use for loop and knn.score to find the best value of k
correctRate = []
for i in range(1,60,2):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train_std, y_train.values)
    correctRate.append(knn.score(X_valid_std, y_valid.values))
    
# Draw the different correct rate in all k
plt.figure(figsize=(10,6))
plt.plot(range(1,60,2),correctRate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Correct Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Correct Rate')

# Predict using KNN
knn_test_score=knn.score(X_test_std,y_test.values)
print('Correct rate using KNN: {:.5f}'.format(knn_test_score))

"""
knn = KNN(X_train_std, X_valid_std, X_test_std, y_train.values, y_valid.values, y_test.values)
knn.findK()
knn.draw()
knn.predict(7)
"""

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
maxRate = 0
indexRate = 0
for i in range (1,30):
    model = DecisionTreeRegressor(random_state = i)
    model.fit(X_train, y_train)
    if maxRate < model.score(X_valid, y_valid):
        maxRate = model.score(X_valid, y_valid)
        indexRate = i
DecisionTreeRegressor(random_state = indexRate)
print("Correct rate using Decision Tree: ", round(model.score(X_test, y_test),5))

# Random Forest
from sklearn.ensemble import RandomForestRegressor
maxRate = 0
indexRate = 0
for i in range (1,10):
    rfc = RandomForestRegressor(random_state = i)
    rfc.fit(X_train, y_train.values.ravel())
    if maxRate < rfc.score(X_valid, y_valid.values.ravel()):
        maxRate = rfc.score(X_valid, y_valid.values.ravel())
        indexRate = i
RandomForestRegressor(random_state = indexRate)
print("Correct rate using Random Forest: ", round(rfc.score(X_test, y_test.values.ravel()),5))

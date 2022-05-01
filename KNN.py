import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

class KNN:
    
    def __init__(self, X_train, X_valid, X_test, y_train, y_valid, y_test):
        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test
        
    def findK(self):
        # Method 1: Use GridSearchCV to find the best value of k
        # 設定需要搜尋的K值，'n_neighbors'是sklearn中KNN的引數
        parameters = {'n_neighbors':[1,3,5,7,9,11,13]}
        knn = KNeighborsRegressor() #注意：這裡不用指定引數
        # 通過GridSearchCV來搜尋最好的K值。這個模組的內部其實就是對每一個K值進行評估
        clf = GridSearchCV(knn,parameters,cv=5) #5折
        clf.fit({self.X_train},{self.y_train})
        # 輸出最好的引數以及對應的準確率
#        print("The best rate is ：%.5f"%clf.best_score_,"The best value of k is",clf.best_params_)
        print("The best value of k is",clf.best_params_)
        
    def draw(self):
        #Method 2: Use for loop and knn.score to find the best value of k
        correctRate = []
        for i in range(1,60,2):
            knn = KNeighborsRegressor(n_neighbors=i)
            knn.fit({self.X_train},{self.y_train})
            correctRate.append(knn.score({self.X_valid},{self.y_valid}))

        # Draw the different correct rate in all k
        plt.figure(figsize=(10,6))
        plt.plot(range(1,60,2),correctRate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
        plt.title('Correct Rate vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Correct Rate')

    def predict(self, k):
        # Predict using KNN
        knn = KNeighborsRegressor(n_neighbors=k)
        score = knn.score({self.X_test}, {self.y_test})
        print('Correct rate using KNN: {:.5f}'.format(score))
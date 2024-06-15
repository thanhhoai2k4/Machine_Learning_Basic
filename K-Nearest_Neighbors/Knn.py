import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from math import sqrt
np.random.seed(10)


# Load data iris
def load_Data_Iris():
    iris = datasets.load_iris()

    iris_X = iris.data
    iris_y = iris.target
    # split dataset to 2 part: train(80%), test(20)
    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, random_state=1, test_size=0.2, shuffle=True)

    return X_train, X_test, y_train, y_test 

# naively compute square distance between two vector
def dist_pc_fast(z, X):
    X2 = np.sum(X*X, 1) # 1 same sa axis = 1
    return X2 - 2*X.dot(z)


X_train, X_test, y_train, y_test = load_Data_Iris()



# a = dist_pc_fast(X_test[0], X_train)


# min = 1000000
# index = -1
# for i in range(len(a)):
#     if (a[i] < min):
#         min = a[i]
#         index = i



# print(y_test[0])
# print(y_train[index])

# predict  value target
def predict(X_train, y_train, z, y):

    # distance each data point in data_test with each data point in data train 
    dist_all = list(dist_pc_fast(z, X_train))

    # get a index in smallest value  
    index_min = dist_all.index(min(dist_all))

    real_Label = y_train[index_min]
    
    predict_label = y

    return real_Label, predict_label

predict(X_train, y_train, X_test[0], y_test[0])

def KNN_train(X_train, y_train, X_test, y_test):
    
    precent = 0
    for i in range(len(X_test)):
        real, pre = predict(X_train, y_train, X_test[i], y_test[i])
        if (real == pre):
            precent += 1

    print("total predict: ", len(X_test))
    print("total right predict: ", precent)
    print("Percent right predict: ", precent/len(X_test)*100 ," %")
    return precent


KNN_train(X_train, y_train, X_test, y_test)
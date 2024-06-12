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


# calculating the distance each data point
def distanceX(X_train):
    distance = np.zeros((X_train.shape[0],1))

    for i in range((X_train.shape[0])):
        distance[i] = np.linalg.norm(X_train[i])
    
    return distance



# Call load_Data_Iris
X_train, X_test, y_train, y_test  = load_Data_Iris()

distanceOfXTrain = distanceX(X_train)
distanceOfXtest  = distanceX (X_test)




b = distanceOfXtest[0]
print(y_test[0])

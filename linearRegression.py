import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
Y = np.array([3,7,5,11,14])

X_b = np.c_[np.ones((X.shape[0], 1)), X]
theta = np.random.randn(X_b.shape[1], 1)

def predict(X, theta):
    return np.dot(X, theta)

def computeCost(X, Y, theta):
    m = len(Y)
    predictions = predict(X, theta)
    cost = (1 / (2*m)) * np.sum((predictions - Y.reshape(-1, 1)) ** 2)
    return cost

def gradientDescent(X, Y, theta, learningRate, iterations):
    m = len(Y)
    costHistory = np.zeros(iterations)
    
    for i in range(iterations):
        gradients = (1/m) * X.T.dot(predict(X, theta) - Y.reshape(-1,1))
        theta = theta - learningRate * gradients
        costHistory[i] = computeCost(X, Y, theta)
        
    return theta, costHistory

learningRate = 0.01
iterations = 1000

thetaOptimal, costHistory = gradientDescent(X_b, Y, theta, learningRate, iterations)

YPred = predict(X_b, thetaOptimal)
print("Predictions: ", YPred)

finalCost = computeCost(X_b, Y, thetaOptimal)
print("Final Cost: ", finalCost)

print("Theta Optimal", thetaOptimal)
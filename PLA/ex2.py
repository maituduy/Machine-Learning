import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import perceptron
from scipy.spatial.distance import cdist

np.random.seed(2)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

d = np.concatenate((X0, X1), axis = 1)

# Labels
t = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Data


colormap = np.array(['r', 'k'])
plt.scatter(d[0], d[1], c=colormap[t], s=40)
#plt.show()
# rotate the data 180 degrees
d90 = np.rot90(d)
d90 = np.rot90(d90)
d90 = np.rot90(d90)

# Create the model
net = perceptron.Perceptron(n_iter=100, eta0=0.002)
net.fit(d90,t)

# Print the results
print("Prediction " + str(net.predict(d90)))
print("Actual     " + str(t))
print("Accuracy   " + str(net.score(d90, t)*100) + "%")
print(d90)
print(t)
# Plot the original data
plt.scatter(d[0], d[1], c=colormap[t], s=40)

# Output the values
print("Coefficient 0 " + str(net.coef_[0,0]))
print("Coefficient 1 " + str(net.coef_[0,1]))
print("Bias " + str(net.intercept_))

# Calc the hyperplane (decision boundary)
ymin, ymax = plt.ylim()
# print(ymin)
# print(ymax)
w = net.coef_[0]
print(net.coef_)
print(net.intercept_)
a = -w[0] / w[1]
xx = np.linspace(ymin, ymax)
yy = a * xx - (net.intercept_[0]) / w[1]

# Plot the line
plt.plot(yy,xx, 'k-')
plt.show()

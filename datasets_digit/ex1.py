from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import perceptron

digits = datasets.load_digits();
label = digits.target
init_data = digits.data
a = [i for i in range(len(label)) if label[i] in [1,2,3,7]]
label = label[a]
data = init_data[a]
# print(init_data)
# print(label)
X_train, X_test, y_train, y_test = train_test_split(data,label,test_size = 0.4)

net = perceptron.Perceptron(n_iter=100, eta0=0.002)
net.fit(X_train,y_train)

data_test = net.predict(X_test)
print(data_test)
print(y_test)
print("Accuracy = ", accuracy_score(data_test,y_test))

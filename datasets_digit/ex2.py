import numpy as np
from mnist import MNIST
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
mntest = MNIST('./MNIST/')

mntest.load_testing()

X_test = np.asarray(mntest.test_images)
y_test = np.asarray(mntest.test_labels)
# a = [i for i in range(len(label)) if label[i] in [0,1,2,3,4,5,6,7,8,9]]
# label = label[a]
# data = data[a]

mntrain = MNIST('./MNIST/')

mntrain.load_training()

X_train = np.asarray(mntrain.train_images)
y_train = np.asarray(mntrain.train_labels)

#X_train, X_test, y_train, y_test = train_test_split(data,label,test_size = 0.3)
#net = perceptron.Perceptron(n_iter=100, eta0=0.002)
# net = mlp = MLPClassifier(verbose=10, learning_rate='adaptive')
# net = SVC(decision_function_shape='ovo',gamma=0.001)
net = LogisticRegression(C=1e5,
        solver = 'lbfgs', multi_class = 'multinomial')
net.fit(X_train,y_train)

data_test = net.predict(X_test)

print("Accuracy = ", accuracy_score(data_test,y_test))

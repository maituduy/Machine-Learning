import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
with open('./letter-recognition.data') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]
labels = []
data = []
for i in content :
    # print(i.strip(','))
    temp = i.split(',')
    labels.append(temp[0])
    data.append(temp[1:])


labels = np.asarray(labels)
new_data = [list(map(int, x)) for x in data]
data = np.asarray(new_data)
X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size = 0.3)

net = mlp = MLPClassifier(verbose=10, learning_rate='adaptive')
net.fit(X_train,y_train)
print(len(labels))
data_test = net.predict(X_test)

# print(len(data_test))
# print(len(y_test))
# print("Accuracy = ", accuracy_score(data_test,y_test))

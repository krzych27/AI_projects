#import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC


data = pd.read_csv('banknotes.txt', sep=",", header=0)
y = data['class']
X = data.drop(['class'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=27)


clf = MLPClassifier(hidden_layer_sizes=(5, 5,5), max_iter=15, alpha=0.0001,
                      activation='logistic', solver='lbfgs', verbose=10, random_state=21, tol=0.00005)


clf = SVC(random_state=52, kernel='rbf')


cm = list()
accuracy = list()
report = list()
#weights = list()
k = 5
kf = KFold(n_splits=k, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred))
    cm.append(confusion_matrix(y_test, y_pred))
    report.append(classification_report(y_test,y_pred))

for x1, x2,x3, i in zip(accuracy, cm,report, range(k)):
    print("Zbiór nr {}".format(i+1))
    print("Wynik klasyfikacji dokładności: {}".format(x1))
    print("\tMacierz błędu: \n{}\n".format(x2))
    print("Podsumowanie klasyfikacji: \n{}\n".format(x3))
   





import sys
import scipy
import numpy
import pandas
#import sklearn
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing, cross_validation

#Load dataset
url = "day.csv"
dataset = pandas.read_csv(url)

# # shape
# print("shape")
#print(dataset.shape)
#
# # head
# print("first 20 data row")
# print(dataset.head(20))


df = dataset[['holiday', 'weekday', 'cnt', 'workingday']]
print(df.shape)

array = df.values
X = array[:,0:2]
y = array[:,3]

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.6)

lr = LinearRegression()
clf = LogisticRegression()
svm = SVC()

lr.fit(X_train, y_train)
clf.fit(X_train, y_train)
svm.fit(X_train, y_train)

lr_accuracy = lr.score(X_test, y_test)
lg_accuracy = clf.score(X_test, y_test)
svm_accuracy = svm.score(X_test, y_test)

print("Linear Regression : ",lr_accuracy*100)
print("Logistic Regression : ",lg_accuracy*100)
print("SVM : ",svm_accuracy*100)

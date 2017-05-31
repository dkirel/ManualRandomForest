import pandas as pd
from sklearn.model_selection import train_test_split

from RandomForest import *

# Weather data
print("Weather data")
data = np.array([['D1',	'Sunny',	'Hot',	'High',	'Weak',	0],
                    ['D2',	'Sunny',	'Hot',	'High',	'Strong',	0],
                    ['D3',	'Overcast',	'Hot',	'High',	'Weak',	1],
                    ['D4',	'Rain',	'Mild',	'High',	'Weak',	1],
                    ['D5',	'Rain',	'Cool',	'Normal',	'Weak',	1],
                    ['D6',	'Rain',	'Cool',	'Normal',	'Strong',	0],
                    ['D7',	'Overcast',	'Cool',	'Normal',	'Strong',	1],
                    ['D8',	'Sunny',	'Mild',	'High',	'Weak',	0],
                    ['D9',	'Sunny',	'Cool',	'Normal',	'Weak',	1],
                    ['D10',	'Rain',	'Mild',	'Normal',	'Weak',	1],
                    ['D11',	'Sunny',	'Mild',	'Normal',	'Strong',	1],
                    ['D12',	'Overcast',	'Mild',	'High',	'Strong',	1],
                    ['D13',	'Overcast',	'Hot',	'Normal',	'Weak',	1],
                    ['D14',	'Rain',	'Mild',	'High',	'Strong',	0]])

X = data[:,:-1]
y = data[:, -1].astype(int)

dt = DecisionTree()
dt.fit(X, y)

accuracy = dt.score(X, y)
print('Simple tree training accuracy: ', accuracy)


# Car data
print("\nCar data")
df = pd.read_csv('data/car.data')
X = df.ix[:, 1:]
y = df.ix[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Decision tree classifier
dt = DecisionTree()
dt.fit(X_train, y_train)

accuracy = dt.score(X_train, y_train)
print('Decision tree training accuracy: ', accuracy)

accuracy = dt.score(X_test, y_test)
print('Decision tree testing accuracy: ', accuracy)

# Random forest classifier
rf = RandomForest(d=4)
rf.fit(X_train, y_train)

accuracy = rf.score(X_train, y_train)
print('Random forest training accuracy: ', accuracy)

accuracy = rf.score(X_test, y_test)
print('Random forest testing accuracy: ', accuracy)


"""
print("\nPruning")
dt.prune(X_test, y_test)
accuracy = dt.score(X_test, y_test)
print('Pruned decision tree accuracy: ', accuracy)
"""

"""
sdt = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
sdt.fit(X_train, y_train)

accuracy = sdt.score(X_test, y_test)
print('Sklearn decision tree accuracy: ', accuracy)
"""

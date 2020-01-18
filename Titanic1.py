import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree, linear_model, model_selection, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import sklearn
sns.set()

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Put the two dataset together to efficiently numerize the string values
train_data_set = [train, test]

# Set the option to show all columns in the dataframe of the Python IDE console
#pd.set_option('display.expand_frame_repr', False)
pd.options.display.width = 0
pd.set_option('display.max_rows', 800)


# The following method is used to 'clean' the data where strings are converted to integer and Nan are filled.
for dataset in train_data_set:
    # changes male gender to 0 and female gender to 1
    dataset.loc[dataset['Sex'] == 'male', 'Sex'] = 0
    dataset.loc[dataset['Sex'] == 'female', 'Sex'] = 1

    # fills Nan value to S and changes the string values to 0, 1, 2 respectively
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset.loc[dataset['Embarked'] == 'S', 'Embarked'] = 0
    dataset.loc[dataset['Embarked'] == 'C', 'Embarked'] = 1
    dataset.loc[dataset['Embarked'] == 'Q', 'Embarked'] = 2

    # instead of removing this 'Name' column, i have changed the title of Mr, Miss, Mrs, etc to various numbers 0, 1, 2, 3 respectively
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    title_mapping = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Dr': 3, 'Rev': 3, 'Col': 3, 'Mlle': 3, 'Major': 3, 'Capt': 3, 'Jonkheer': 3, 'Ms': 3, 'Lady': 3, 'Sir': 3, 'Countess': 3, 'Mme': 3, 'Don': 3, 'Dona': 3}
    dataset['Title'] = dataset['Title'].map(title_mapping)

    # fills the Nan value of age to the median value of the various 'titles'
    dataset['Age'].fillna(dataset.groupby('Title')['Age'].transform('median'), inplace=True)
    dataset['Age'] = preprocessing.scale(dataset['Age'])

    # fills the Nan value of Fare to the median value based on their Pclass
    dataset['Fare'].fillna(dataset.groupby('Pclass')['Fare'].transform('median'), inplace=True)
    dataset['Fare'] = preprocessing.scale(dataset['Fare'])

    # Drops the 'Name' column
    dataset.drop('Name', axis = 1, inplace= True)

    # Drops the 'Ticket' column
    dataset.drop('Ticket', axis = 1, inplace=True)

    # Drops the 'Cabin' column
    #dataset.drop('Cabin', axis = 1, inplace=True)
    dataset['Cabin'] = dataset['Cabin'].str[:1]
    cabin_mapping = {'A': 0, 'B': 0.4, 'C': 0.8, 'D': 1.2, 'E': 1.6, 'F': 2.0, 'G': 2.4, 'H':2.8}
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
    dataset['Cabin'].fillna(dataset.groupby('Pclass')['Cabin'].transform('median'), inplace=True)

# print(train.head(100))
"""
# Linear Regression
predict = 'Survived'
X = train.drop(['Survived'], 1).values
y = train[predict].values

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)

print(acc)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
"""


target = train['Survived'].values
features = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'Title']].values

# Logistic Regression
def logistic_regression(features, target):
    logistic = linear_model.LogisticRegression()
    logistic.fit(features, target)
    acc1 = logistic.score(features, target)
    print('Logistic Regression:', acc1)

    scores1 = model_selection.cross_val_score(logistic, features, target, scoring='accuracy', cv=10)
    print(scores1.mean())

    return logistic

# Logistic Regression with polynomial n=2
poly = preprocessing.PolynomialFeatures(degree=2)
features_ = poly.fit_transform(features)
logistic2 = linear_model.LogisticRegression(C=10, max_iter=1000)
logistic2.fit(features_, target)
acc12 = logistic2.score(features_, target)
print('Logistic Regression with Poly(2):', acc12)

scores1 = model_selection.cross_val_score(logistic2, features_, target, scoring='accuracy', cv=10)
print(scores1.mean())

# Decision Tree
def decision_tree(features, target):
    decision = tree.DecisionTreeClassifier()
    decision.fit(features, target)
    acc2 = decision.score(features, target)
    print('Decision Tree:', acc2)

    scores2 = model_selection.cross_val_score(decision, features, target, scoring='accuracy', cv=10)
    print(scores2.mean())
    return decision
# Random Forest
def random_forest(features, target):
    randomForest = RandomForestClassifier(n_estimators=12)
    randomForest.fit(features, target)
    acc3 = randomForest.score(features, target)
    print('Random Forest:', acc3)

    scores3 = model_selection.cross_val_score(randomForest, features, target, scoring='accuracy', cv=10)
    print(scores3.mean())
    return randomForest
# Support Vector Machine
def svm(features, target):
    clf = SVC()
    clf.fit(features, target)
    acc4 = clf.score(features, target)
    print('SVM:', acc4)

    scores4 = model_selection.cross_val_score(clf, features, target, scoring='accuracy', cv=10)
    print(scores4.mean())
    return clf
# KNN
def knn(features, target):
    knntest = KNeighborsClassifier(n_neighbors=13)
    knntest.fit(features, target)
    acc5 = knntest.score(features, target)
    print('KNN:', acc5)

    scores5 = model_selection.cross_val_score(knntest, features, target, scoring='accuracy', cv=10)
    print(scores5.mean())
    return knntest

logistic_regression(features, target)
decision_tree(features, target)
random_forest(features, target)
svm(features, target)
knn(features, target)

#test_features = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'Title']].values
#test_features_ = poly.fit_transform(test_features)
#predict = logistic2.predict(test_features_)

"""
def final(prediction, test):
    submission = pd.DataFrame({
        'PassengerId' : test['PassengerId'],
        'Survived' : prediction
    })
    submission.to_csv('submission.csv', index=False)
"""
#final(predict, test)



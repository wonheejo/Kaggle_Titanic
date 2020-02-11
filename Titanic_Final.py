import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

pd.options.display.width = 0
pd.set_option('display.max_rows', 800)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Fill in missing for Embarked and Fare
train['Embarked'] = train['Embarked'].fillna('S')
test['Fare'] = test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'))

# Fill in the missing age based on the sex, pclass and age
age_male_train = round(train[(train['Sex'] == 'male')]['Age'].groupby(train['Pclass']).mean(), 2)
age_female_train = round(train[(train['Sex'] == 'female')]['Age'].groupby(train['Pclass']).mean(), 2)
age_male_test = round(test[(test['Sex'] == 'male')]['Age'].groupby(test['Pclass']).mean(), 2)
age_female_test = round(test[(test['Sex'] == 'female')]['Age'].groupby(test['Pclass']).mean(), 2)

age_model = pd.concat([age_male_train, age_female_train, age_male_test, age_female_test], axis=1,
	keys=['Train Male', 'Train Female', 'Test Male', 'Test Female'])
# print(age_model)

def fillna_age_train(APStrain):
	Age = APStrain[0]
	Pclass = APStrain[1]
	Sex = APStrain[2]

	if pd.isnull(Age):
		if Sex == 'male':
			if Pclass == 1:
				return 41.28
			if Pclass == 2:
				return 30.74
			if Pclass == 3:
				return 26.51
		if Sex == 'female':
			if Pclass == 1:
				return 34.61
			if Pclass == 2:
				return 28.72
			if Pclass == 3:
				return 21.75

	else:
		return Age

def fillna_age_test(APStest):
	Age = APStest[0]
	Pclass = APStest[1]
	Sex = APStest[2]

	if pd.isnull(Age):
		if Sex == 'male':
			if Pclass == 1:
				return 40.52
			if Pclass == 2:
				return 30.94
			if Pclass == 3:
				return 24.53
		if Sex == 'female':
			if Pclass == 1:
				return 41.33
			if Pclass == 2:
				return 24.38
			if Pclass == 3:
				return 23.07

	else:
		return Age

train['Age'] = train[['Age', 'Pclass', 'Sex']].apply(fillna_age_train, axis=1)
test['Age'] = test[['Age', 'Pclass', 'Sex']].apply(fillna_age_test, axis=1)

"""
for cols in train.columns.tolist():
	if train[cols].isnull().sum():
		print('Train Set = Missing {}: {}/{}' .format(cols, train[cols].isnull().sum(), train.shape[0]))

for cols in test.columns.tolist():
	if test[cols].isnull().sum():
		print('Test Set = Missing {}: {}/{}' .format(cols, test[cols].isnull().sum(), test.shape[0]))
"""

train_test_set = [train, test]

# Feature Engineering
for dataset in train_test_set:

	# Sex of male into 0 and 1
	dataset.loc[dataset['Sex'] == 'male', 'Sex'] = 0
	dataset.loc[dataset['Sex'] == 'female', 'Sex'] = 1

	# Embarked status into 0, 1, 2
	dataset.loc[dataset['Embarked'] == 'S', 'Embarked'] = 0
	dataset.loc[dataset['Embarked'] == 'C', 'Embarked'] = 1
	dataset.loc[dataset['Embarked'] == 'Q', 'Embarked'] = 2

	# Replacing Names into title based on the string before '.'
	dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)
	title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Dr': 4, 'Rev': 4, 'Mlle': 4, 'Major': 4, 'Col': 4, 'Lady': 4, 'Capt': 4, 'Ms': 4, 'Sir': 4, 'Jonkheer': 4, 'Mme': 4, 'Countess': 4, 'Don': 4, 'Dona': 4}
	dataset['Title'] = dataset['Title'].map(title_map)

	# Data binning of Age column
	dataset.loc[dataset['Age'] <= 16.33, 'Age'] = 0
	dataset.loc[(dataset['Age'] > 16.33) & (dataset['Age'] <= 32.25), 'Age'] = 1
	dataset.loc[(dataset['Age'] > 32.25) & (dataset['Age'] <= 48.17), 'Age'] = 2
	dataset.loc[(dataset['Age'] > 48.17) & (dataset['Age'] <= 64.08), 'Age'] = 3
	dataset.loc[dataset['Age'] > 64.08, 'Age'] = 4

	# Data binning of Fare column
	dataset.loc[dataset['Fare'] <= 7.85, 'Fare'] = 0
	dataset.loc[(dataset['Fare'] > 7.85) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
	dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.68), 'Fare'] = 2
	dataset.loc[(dataset['Fare'] > 21.68) & (dataset['Fare'] <= 39.69), 'Fare'] = 3
	dataset.loc[dataset['Fare'] > 39.69, 'Fare'] = 4

	# Data binning of Family column
	dataset['Family'] = dataset['SibSp'] + dataset['Parch']
	dataset.loc[dataset['Family'] == 0, 'Family'] = 0
	dataset.loc[(dataset['Family'] > 0) & (dataset['Family'] <= 3), 'Family'] = 1
	dataset.loc[(dataset['Family'] > 3) & (dataset['Family'] <= 5), 'Family'] = 2
	dataset.loc[(dataset['Family'] > 5) & (dataset['Family'] <= 6), 'Family'] = 3
	dataset.loc[dataset['Family'] > 6, 'Family'] = 4

	dataset['Single']= dataset['Family'].map(lambda s: 1 if s==0 else 0)
	dataset['SmFm'] = dataset['Family'].map(lambda s: 1 if s==1 else 0)
	dataset['MdFm'] = dataset['Family'].map(lambda s: 1 if s==2 else 0)
	dataset['LgFm'] = dataset['Family'].map(lambda s: 1 if s==3 else 0)
	dataset['VlgFm'] = dataset['Family'].map(lambda s: 1 if s==4 else 0)

train = train.drop(['Name', 'SibSp', 'Parch', 'Cabin', 'Ticket', 'Family'], axis=1)
test = test.drop(['Name', 'SibSp', 'Parch', 'Cabin', 'Ticket', 'Family'], axis=1)

# print(dataset['Title'].value_counts())

x_train = train.drop(['PassengerId', 'Survived'], axis=1)
y_train = train['Survived']
x_test = test.drop(['PassengerId'], axis=1)
# print(x_train.shape, y_train.shape, x_test.shape)

# SGDClassifier
sgd_clf = SGDClassifier()
sgd_clf.fit(x_train, y_train)
sgd_clf_cvs = cross_val_score(sgd_clf, x_train, y_train, cv=10, scoring='accuracy').mean()
#sgd_clf_pre = cross_val_predict(sgd_clf, x_train, y_train, cv=10)
#sgd_f1 = f1_score(y_train, sgd_clf_pre)


# Logistic Regression
lg_clf = LogisticRegression()
lg_clf.fit(x_train, y_train)
lg_clf_cvs = cross_val_score(lg_clf, x_train, y_train, cv=10, scoring='accuracy').mean()
#lg_clf_pre = cross_val_predict(lg_clf, x_train, y_train, cv=10)
#lg_f1 = f1_score(y_train, lg_clf_pre)


# SVC
svm_clf = SVC(probability=True)
svm_clf.fit(x_train, y_train)
svm_clf_cvs = cross_val_score(svm_clf, x_train, y_train, cv=10, scoring='accuracy').mean()
#svm_clf_pre = cross_val_predict(svm_clf, x_train, y_train, cv=10)
#svm_f1 = f1_score(y_train, svm_clf_pre)


# Decision Tree Classifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(x_train, y_train)
dt_clf_cvs = cross_val_score(dt_clf, x_train, y_train, cv=10, scoring='accuracy').mean()
#dt_clf_pre = cross_val_predict(dt_clf, x_train, y_train, cv=10)
#dt_f1 = f1_score(y_train, dt_clf_pre)


# KNNeighbors Classifier
knn_clf = KNeighborsClassifier(n_neighbors = 3)
knn_clf.fit(x_train, y_train)
knn_clf_cvs = cross_val_score(knn_clf, x_train, y_train, cv=10, scoring='accuracy').mean()
#knn_clf_pre = cross_val_predict(knn_clf, x_train, y_train, cv=10)
#knn_f1 = f1_score(y_train, knn_clf_pre)


# RandomForest Classifier
rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rf_clf.fit(x_train, y_train)
rf_clf_cvs = cross_val_score(rf_clf, x_train, y_train, cv=10, scoring='accuracy').mean()
#rf_clf_pre = cross_val_predict(rf_clf, x_train, y_train, cv=10)
#rf_f1 = f1_score(y_train, rf_clf_pre)

"""
bag_clf = BaggingClassifier(
	DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True, n_jobs=-1, oob_score=True)
bag_clf.fit(x_train, y_train)
bag_clf_cvs = cross_val_score(bag_clf, x_train, y_train, cv=10, scoring='accuracy').mean()
bag_clf_pre = cross_val_predict(bag_clf, x_train, y_train, cv=10)
bag_f1 = f1_score(y_train, bag_clf_pre)
"""

voting_clf = VotingClassifier(
	estimators=[('lr', lg_clf), ('rf', rf_clf), ('svc', svm_clf)],
	voting='soft')
voting_clf.fit(x_train, y_train)
"""
xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(x_train, y_train, eval_set=[(X_val, Y_val)], early_stopping_rounds=2)
xgb_score = xgb_reg.score(x_train, y_train)
y_pred = xgb_reg.predict(X_val)
print(xgb_score)
print(y_pred)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=0.1)
gbrt.fit(x_train, y_train)
print(gbrt.score(x_train, y_train))


for clf in (lg_clf, svm_clf, dt_clf, rf_clf, voting_clf, bag_clf):
	clf.fit(x_train, y_train)
	y_score = clf.score(x_train, y_train)
	y_cvs = cross_val_score(clf, x_train, y_train, cv=10, scoring='accuracy').mean()
	print(clf.__class__.__name__, y_cvs)

models = pd.DataFrame({
	'Models': ['SGDClassifier', 'LogisticRegression', 'SVM', 'DecisionTree', 'KNNeighbors', 'RandomForest', 'Voting', 'Bagging'],
	'CrossValScore': [sgd_clf_cvs, lg_clf_cvs, svm_clf_cvs, dt_clf_cvs, knn_clf_cvs, rf_clf_cvs, voting_clf_cvs, bag_clf_cvs],
	'F1 Score': [sgd_f1, lg_f1, svm_f1, dt_f1, knn_f1, rf_f1, voting_f1, bag_f1]
	})

models.sort_values(by='Models', ascending=False)
print(models)


print('\n')
#print(sgd_dfscore)
"""

predict = voting_clf.predict(x_test)


def final(predict, test):
    submission = pd.DataFrame({
        'PassengerId' : test['PassengerId'],
        'Survived' : predict
    })
    submission.to_csv('submissionVotingClassifier.csv', index=False)

final(predict, test)




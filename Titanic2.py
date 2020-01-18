import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

female_color = '#FA0000'

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Visualizing the data in various ways to understand it
fig = plt.figure(figsize=(18, 6))

plt.subplot2grid((3, 4), (0, 0))
train.Survived.value_counts(normalize=True).plot(kind='bar', alpha=0.5)
plt.title("Survived")

plt.subplot2grid((3, 4), (0, 1))
train.Survived[train.Sex == 'male'].value_counts(normalize=True).plot(kind='bar', alpha=0.5)
plt.title("Survived wrt Male")

plt.subplot2grid((3, 4), (0, 2))
train.Survived[train.Sex == 'female'].value_counts(normalize=True).plot(kind='bar', alpha=0.5, color=female_color)
plt.title("Survived wrt Female")

plt.subplot2grid((3, 4), (0, 3))
train.Sex[train.Survived == 1].value_counts(normalize=True).plot(kind='bar', alpha=0.5, color=[female_color, 'b'])
plt.title("Survived wrt Female")

plt.subplot2grid((3, 4), (1, 0), colspan=4)
for x in [1, 2, 3]:
    train.Survived[train.Pclass == x].plot(kind='kde')
plt.title('Class with Survived')
plt.legend(('1st', '2nd', '3rd'))

plt.subplot2grid((3, 4), (2, 0))
train.Survived[(train.Sex == 'male') & (train.Pclass == 1)].value_counts(normalize=True).plot(kind='bar', alpha=0.5, color=[female_color, 'b'])
plt.title("Survived wrt rich male")

plt.subplot2grid((3, 4), (2, 1))
train.Survived[(train.Sex == 'male') & (train.Pclass == 3)].value_counts(normalize=True).plot(kind='bar', alpha=0.5, color=[female_color, 'b'])
plt.title("Survived wrt poor male")

plt.show()


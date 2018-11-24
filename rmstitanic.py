
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train = train.drop(['Cabin'], axis = 1)#because there is so much missing data from cabin just remove it
train['Embarked'] = train['Embarked'].fillna('S') #so can just replace with the most common value which is S
for pclass in train['Pclass'].unique(): #for every unique pclass, locate the age column value, match to groupe by Pclass means
    train.loc[train['Pclass'] == pclass, 'Age'] = train.groupby('Pclass')['Age'].mean().loc[pclass]


print (train.head())
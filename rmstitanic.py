import pandas as pd

def cleandata(df):
	df = train.drop(['Cabin'], axis = 1)#because there is so much missing data from cabin just remove it
	df['Embarked'] = train['Embarked'].fillna('S') #so can just replace with the most common value which is S
	for pclass in df['Pclass'].unique(): #for every unique pclass, locate the age column value, match to groupe by Pclass means
	    df.loc[df['Pclass'] == pclass, 'Age'] = df.groupby('Pclass')['Age'].mean().loc[pclass]
	return df


if __name__ == 'main':
    train = pd.read_csv('train.csv')
    train = cleandata(train)
    print (train.head())
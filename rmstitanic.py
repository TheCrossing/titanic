
# coding: utf-8

# In[277]:


import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[248]:


#Edward J Smith - The captain goes doen with the ship
#we are trying to predict whether a passanger would survive using the train test
#then test it on the testing set


# In[280]:



test.head()
X = train
train.head() #feature matrix


# In[281]:


X.isnull().sum().astype(float) /X.shape[0]#check feature matrix for nas
#I have nas so I need to either fill them in with sume value or remove them from the dataset
X = X.drop(['Cabin'], axis = 1)#because there is so much missing data from cabin just remove it


# In[282]:


X['Embarked'].value_counts() #only 2 values from embarked are missing, options are S,C or Q so can just use S, most common
X['Embarked'] = X['Embarked'].fillna('S') #so can just replace with the most common value which is S


# In[283]:



import matplotlib.pyplot as plt
X.isnull().sum().astype(float) /X.shape[0] #only column with nas now is Age
 #get distribution of Age
X['Age'].hist()
X['Age'].mean() #one option is to fill in the nas with the mean (29)
X['Age'].median() #(28) so data is slightly skewed to the right


# In[284]:


#compare age versus pclass too see how age an pclass fit together
import seaborn as sns
for pclass in X['Pclass'].unique(): #for every unique pclass
    sns.kdeplot(X[X['Pclass'] == pclass]['Age'], label = pclass)#groupy pclass and density plot for age
    #we see a trend of the older the person the higher the class (1 = highest)
    


# In[285]:


#so to replace missing age nas values we might group by pclass and assign mean age for each class
X.groupby('Pclass')['Age'].mean()

for pclass in X['Pclass'].unique(): #for every unique pclass, locate the age column value, match to groupe by Pclass means
    X.loc[(X['Pclass'] == pclass) & (X['Age'].isnull()),'Age'] = X.groupby('Pclass')['Age'].mean().loc[pclass]


# In[287]:


X.isnull().sum() # wehave finally got rid of all the missing data
#X.head() #all age values are now means grouped by Pclass
#now we can assess which variables are important when considering passenger survival.
# [Feature Engineering and transformations]


# In[157]:


#visual on survival and Pclass to determine if Pclass is a good predictor
plt.figure(figsize= [16,8])
plt.subplot(121)
plt.hist(train[X['Survived'] == 0]['Pclass'])
plt.title("Perished")
plt.xlabel("Ticket Class")
plt.subplot(122)
plt.hist(train[X['Survived'] == 1]['Pclass'])
plt.title("Survived")
plt.xlabel("Ticket Class")
   


# In[158]:


#to get better understanding you might want the mean as well
X.groupby('Pclass')['Survived'].mean()
#so this is obviously going to be a pretty good predictor features of who would survive, PClass


# In[145]:


#visual on Age and Pclass to determine if Pclass is a good predictor
#turns out Age is not a great predictor of survival, although there is the spike for infants among survivors
#versus perished so knowing whether or not they are an infant would be helpful
print (X.groupby('Survived')['Age'].mean())
plt.figure(figsize= [8,4])
plt.subplot(121)
plt.hist(train[X['Survived'] == 0]['Age'])
plt.title("Perished")
plt.xlabel("Age of Passenger")
plt.subplot(122)
plt.hist(train[X['Survived'] == 1]['Age'])
plt.title("Survived")
plt.xlabel("Age of Passenger")


# In[129]:


#create an infant variable to use as a predictor
#an idea to try later, distribution if age older than 10 for survivor and perish groups looks quite similar
X['Infant'] = X['Age'] < 10
X.head()


# In[288]:


# How to do consider string data and categorical data
#you could make categorical data numeric, e.g Sex could be 1,0 or Embarked could be 1,2,3 but numerical data could interpret 
# embarked 3 being much more different to 1 than it is from 2, the ML estimator will interpret these that way
#as feature matrix only can have numerical values for scikit learn then they must be changed 

X['Sex'].replace({'male': 0, 'female':1}, inplace = True) #create dictionary to replace sex data


# In[289]:


#Ticket data appears random and not useful for any predictions so remove it
X.drop(['Ticket'], axis = 1, inplace = True)


# In[303]:


#consider embarked feature with one hot encoding
#this means that instead of having 1 embarked column we would have 3
X['Embarked_S'] = 0
X['Embarked_C'] = 0
X['Embarked_Q'] = 0
X.loc[X['Embarked'] == 'S', 'Embarked_S'] = 1
X.loc[X['Embarked'] == 'C', 'Embarked_C'] = 1
X.loc[X['Embarked'] == 'Q', 'Embarked_Q'] = 1
X.drop(['Embarked'], axis = 1, inplace = True)


# In[291]:


#Now to consider the name column
#Mrs,ms,Masters etc...is it a predictor versus Mr
#take name feature, split on , and get 1st value, split again on . and get 0th value, then get all unique values
X['Name'].str.split(',').apply(lambda x: x[1]).str.split('.').apply(lambda x: x[0]).unique()
#an idea for later to see if it is a good predictor when having a royal title etc

#for now
X.drop(['Name'], axis = 1, inplace = True)


# In[292]:


X.drop(['PassengerId'], axis = 1, inplace = True) #dont need passengerid as it is random 
#so not a good predictor


# In[304]:


#y = X.pop('Survived') #this is my target label data
X.head() #is my feature matrix
#y.head()


# In[293]:


#building the model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


# In[305]:


model = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression()),
    ])

#gridsearch for feature engineering, look at hyper-parameters of Logistic Regression
gs = GridSearchCV(
    model,
    {'logreg__penalty':['l1','l2'],
    'logreg__C':[0.01, 0.1, 1, 10, 100]},
    cv = 5,
    n_jobs = 4    
)


# In[306]:


gs.fit(X,y)


# In[307]:


gs.best_params_ #tells us which hyper-paramters performed best
#so I can go back and just enter these parameters into the hyper parameters under logistic regression model(penalty = etc)


# In[308]:


test.head()


# In[309]:


#now need to do the same data munging to our test data (target label data)
test = test.drop(['Cabin'], axis = 1)#because there is so much missing data from cabin just remove it
test.drop(['Name'], axis = 1, inplace = True)
test.drop(['Ticket'], axis = 1, inplace = True)
passengertestID = test.pop('PassengerId') #pop passenger ID to add bacl into test df later, for predict probabilitys
#need the shape of test and train data to be the same


# In[310]:


test['Embarked'] = test['Embarked'].fillna('S')
for pclass in test['Pclass'].unique(): #for evertest unique pclass, locate the age column value, match to groupe btest Pclass means
    test.loc[(test['Pclass'] == pclass) & (test['Age'].isnull()),'Age'] = test.groupby('Pclass')['Age'].mean().loc[pclass]
test['Sex'].replace({'male': 0, 'female':1}, inplace = True) #create dictionartest to replace setest data

test['Embarked_S'] = 0
test['Embarked_C'] = 0
test['Embarked_Q'] = 0
test.loc[test['Embarked'] == 'S', 'Embarked_S'] = 1
test.loc[test['Embarked'] == 'C', 'Embarked_C'] = 1
test.loc[test['Embarked'] == 'Q', 'Embarked_Q'] = 1
test.drop(['Embarked'], axis = 1, inplace = True)


# In[313]:


test['Fare'] = test['Fare'].fillna('9.0')
test.isnull().sum().astype(float) /test.shape[0] #clear test nans


# In[314]:


score = gs.score(X, y)
print('Logistic Regression pipeline test accuracy: %.3f' % score)


# In[315]:


predictions = gs.predict_proba(test)
default_prob = [i[1] for i in predictions] #Probability for survival


# In[316]:


#create dataframe for predicitons
#dont have the correct passenger ids
prediction = gs.predict(test)
survived = pd.DataFrame({'Survived': prediction}) #prediction of survival generated by model for each test passenger row
survived['PassengerID'] = passengertestID #add correct passenger ids back from test


# In[318]:


survived.head()
survived.to_csv('results.csv', encoding='utf-8', index=False)


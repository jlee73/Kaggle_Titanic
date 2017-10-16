# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 15:27:04 2017

Kaggle_Titanic_datasets using Logistic Regression

@author: James Lee
"""
#import package
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv('../TitanicProject/train.csv')
test_df = pd.read_csv('../TitanicProject/test.csv')
print(train_df.columns.values)


#data preprocessing

#Categorical: Survived Sex Embarked
#Ordinal: Pclass
#Cardinal: Age Fare Discrete SibSp Parch
#print(train_df.head())
print(train_df.tail())
train_df.info()
print('_'*40)
test_df.info()
print(train_df.describe())
print(train_df.describe(include=['O']))


# 'Cabin': High missing rate
# 'Ticket': No correlation 

# Drop 'Cabin', 'Ticket', 'Name', 'PassengerId' in training data
train_df = train_df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)
# Drop 'Cabin', 'Ticket', 'Name' in test data
test_df = test_df.drop(['Cabin', 'Ticket', 'Name'], axis=1)
combine = [train_df, test_df]

#missing value on 'Embarked' is replaced by the most frequent value
#Change Alphanumerical var to Categorical numbers
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    dataset['Sex']=dataset['Sex'].map({'female':1,'male':0}).astype(int)
    dataset['Embarked']=dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)

# using mean value of age on each Pclass and Sex to substitute missing value on Age(714/891)
guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.mean()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

#print(guess_ages)
train_df.head()
print(train_df.head())

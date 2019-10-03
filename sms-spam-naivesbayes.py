# -*- coding: utf-8 -*-
"""
Created on Thu Aug 9 09:37:50 2019

@author: Divisha
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


#import the dataset
df = pd.read_table('SMSSpamCollection', sep='\t', names= ['label', 'message'])
df.head()

#converting labels to binary variables
df['label'] = df.label.map({'ham':0, 'spam':1})

#splitting data into training set and tests set
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], random_state=1)

#apply BoW to dataset using sklearn
#Instantiate the CountVertorizer method
count_vector = CountVectorizer()
#Fit the training data and return the matrix
training_data = count_vector.fit_transform(X_train)
#transfrom the testing data and return the matrix 
testing_data = count_vector.transform(X_test)

#naive bayes classifier with sklearn
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

#making predictions
predictions = naive_bayes.predict(testing_data)

#evaluation
print('Accuracy score:', format(accuracy_score(y_test, predictions)))
print('Precision score:', format(precision_score(y_test, predictions)))
print('Recall score:', format(recall_score(y_test, predictions)))
print('F1 score:', format(f1_score(y_test, predictions)))



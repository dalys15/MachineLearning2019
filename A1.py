#!/usr/bin/env python
# coding: utf-8

# In[43]:


from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV, BayesianRidge, LassoLars
from sklearn.svm import LinearSVR


# In[11]:


#Functions

def normalize(data):
    data = data-data.mean()/data.std()#(data-data.min()/(data.max()-data.min()))
    return data


def cleanData(text_to_clean):
    #get rid of nulls
    text_to_clean['Year of Record'] = text_to_clean['Year of Record'].replace(np.nan, 0)
    text_to_clean['Age'] = text_to_clean['Age'].replace(np.nan, 0)
    text_to_clean['Gender'] = text_to_clean['Gender'].replace("other", '0')
    text_to_clean['Gender'] = text_to_clean['Gender'].replace("unknown", '0')
    text_to_clean['Gender'] = text_to_clean['Gender'].replace(np.nan, '0')
    text_to_clean['Profession'] = text_to_clean['Profession'].replace(np.nan, '0')
    text_to_clean['University Degree'] = text_to_clean['University Degree'].replace(np.nan, '0')
    
    #split gender into 3 varibles -> Gender_female, Gender_male and Gender_0
    nominal_columns = ["Gender"]
    dummy_df = pd.get_dummies(text_to_clean[nominal_columns])
    text_to_clean = pd.concat([text_to_clean, dummy_df], axis=1)
    text_to_clean = text_to_clean.drop(nominal_columns, axis=1)
    
    #drop hair colour
    text_to_clean = text_to_clean.drop(columns='Hair Color')
    
    #Order qualifications
    mapping_dict = {
    "University Degree": 
        {
            "0": 1,
            "No": 1,
            "Bachelor": 2,
            "Master": 3,
            "PhD": 4
        }
    }
    
    text_to_clean = text_to_clean.replace(mapping_dict)
    
    #normalize data
    text_to_clean['Year of Record'] = normalize(text_to_clean['Year of Record'])
    text_to_clean['Age'] = normalize(text_to_clean['Age'])
    text_to_clean['Size of City'] = normalize(text_to_clean['Size of City'])
    text_to_clean['University Degree'] = normalize(text_to_clean['University Degree'])
    text_to_clean['Body Height [cm]'] = normalize(text_to_clean['Body Height [cm]'])
    
    #remove spaces from profession and country
    #text_to_clean.loc[:,'Profession'] = text_to_clean.loc[:,'Profession'].str.replace(' ', '')
    #text_to_clean.loc[:,'Country'] = text_to_clean.loc[:,'Country'].str.replace(' ', '')
    
    return text_to_clean

def dfToArray(data):
    data_array = data.values
    return data_array

#transformer,
def TFIDFscores_train(vectorizer,  data_array): 
    count_vector = vectorizer.fit_transform(data_array)
    #tfidf_scores = transformer.fit_transform(count_vector.toarray())
    #df_tfidf = pd.DataFrame(tfidf_scores.toarray())
    #count_vector = normalize(count_vector.toarray())
    df_tfidf = pd.DataFrame(count_vector.toarray())
    return df_tfidf

def TFIDFscores_test(vectorizer, data_array):
    count_vector = vectorizer.transform(data_array)
    #tfidf_scores = transformer.transform(count_vector.toarray())
    #df_tfidf = pd.DataFrame(tfidf_scores.toarray())
    #count_vector = normalize(count_vector.toarray())
    df_tfidf = pd.DataFrame(count_vector.toarray())
    return df_tfidf


# In[70]:


#clean data
total_data = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv', low_memory=False)
unknownIncome_data = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv', low_memory=False)
unknownIncome_data = unknownIncome_data.drop(columns='Income')

total_data = cleanData(total_data)
unknownIncome_data = cleanData(unknownIncome_data)


# In[71]:


#split training and test data
df_X = total_data.drop(columns='Income in EUR')
df_Y = total_data['Income in EUR']

X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=0.3)


# In[72]:


#get TFIDF scores --> you could probs make another function but sure
professions_train = dfToArray(X_train['Profession'])
professions_test = dfToArray(X_test['Profession'])
professions_unknownIncome = dfToArray(unknownIncome_data['Profession'])

countries_train = dfToArray(X_train['Country'])
countries_test = dfToArray(X_test['Country'])
countries_unknownIncome = dfToArray(unknownIncome_data['Country'])

profession_vectorizer = TfidfVectorizer()
countries_vectorizer = TfidfVectorizer()


#TRAINING
df_profession_train = TFIDFscores_train(profession_vectorizer,  professions_train)
df_countries_train = TFIDFscores_train(countries_vectorizer, countries_train)

instance = dfToArray(X_train['Instance'])

df_profession_train['Instance'] = instance
df_countries_train['Instance'] = instance
X_train = pd.merge(X_train, df_profession_train, how='left', on=['Instance'])
X_train = pd.merge(X_train, df_countries_train, how='left', on=['Instance'])

X_train = X_train.drop(columns=['Instance', 'Profession', 'Country'], axis=1)

#TESTING
df_profession_test = TFIDFscores_test(profession_vectorizer,  professions_test)
df_countries_test = TFIDFscores_test(countries_vectorizer, countries_test)

instance = dfToArray(X_test['Instance'])

df_profession_test['Instance'] = instance
df_countries_test['Instance'] = instance

X_test = pd.merge(X_test, df_profession_test, how='left', on=['Instance'])
X_test = pd.merge(X_test, df_countries_test, how='left', on=['Instance'])

X_test = X_test.drop(columns=['Instance', 'Profession', 'Country'], axis=1)

#UNKOWN INCOME DATA
df_profession_unknownIncome = TFIDFscores_test(profession_vectorizer, professions_unknownIncome)
df_countries_unknownIncome = TFIDFscores_test(countries_vectorizer, countries_unknownIncome)

instance = dfToArray(unknownIncome_data['Instance'])

df_profession_unknownIncome['Instance'] = instance
df_countries_unknownIncome['Instance'] = instance

unknownIncome_data = pd.merge(unknownIncome_data, df_profession_unknownIncome, how='left', on=['Instance'])
unknownIncome_data = pd.merge(unknownIncome_data, df_countries_unknownIncome, how='left', on=['Instance'])

unknownIncome_data = unknownIncome_data.drop(columns=['Instance', 'Profession', 'Country'], axis=1)


# In[73]:


print('X_train:', X_train.shape)
print('\nX_test:', X_test.shape)
print('\nUnknown income data:', unknownIncome_data.shape)


# In[79]:


#make a model
regressor = Ridge() ## 81

regressor.fit(X_train, Y_train)
regressor.score(X_train, Y_train)
predictions = regressor.predict(X_test)
print('Accuracy:', regressor.score(X_test, Y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, predictions))
print('Mean squared Error:', metrics.mean_squared_error(Y_test, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))


# In[51]:


income = regressor.predict(unknownIncome_data)


# In[54]:


print(income.shape)
test = pd.DataFrame(income)
print(test.shape)


# In[55]:


file = open('Ridge.csv', 'w', newline='')
test.to_csv(file)


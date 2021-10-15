#!/usr/bin/env python
# coding: utf-8

# In[4]:


# The aim of this part is creating a module for absenteeism project. 
# This allows us to use the saved model and scaler to test new data

# Basically this is the code we used in absenteeism project updated to a class form

# At the end we write it as .py file (File -> Download as -> Python (.py))


# In[1]:


# Importing libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


# In[2]:


# Creating custom scaler class
class CustomScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.scaler = StandardScaler()
        self.columns = columns
        self.mean_ = None
        self.var_ = None
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# In[3]:


# Creating class to predict on new data
class absenteeism_model():
    
    # Reading files
    def __init__(self,model_file, scaler_file):
        with open('model', 'rb') as model_file, open('scaler', 'rb') as scaler_file:
            self.model = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
            
    # Preprocessing new data
    def load_and_clean_data(self, data_file):
        df = pd.read_csv(data_file, delimiter=',')
        self.dt_with_predictions = df.copy()
        df = df.drop(['ID'], axis = 1)
        df['Absenteeism Time in Hours'] = 'NaN'
        
        rfa_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)
        
        rfa_group_1 = rfa_columns.loc[:,'1':'14'].max(axis=1)
        rfa_group_2 = rfa_columns.loc[:,'15':'17'].max(axis=1)
        rfa_group_3 = rfa_columns.loc[:,'18':'21'].max(axis=1)
        rfa_group_4 = rfa_columns.loc[:,'22':].max(axis=1)
        
        df = df.drop(['Reason for Absence'], axis = 1)
        
        df = pd.concat([df, rfa_group_1, rfa_group_2, rfa_group_3, rfa_group_4], axis = 1)
        
        column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age','Daily Work Load Average',
                        'Body Mass Index', 'Education','Children', 'Pets', 'Absenteeism Time in Hours', 
                        'Rfa_group_1', 'Rfa_group_2', 'Rfa_group_3', 'Rfa_group_4']
        
        df.columns = column_names
        
        columns_reordered = ['Rfa_group_1', 'Rfa_group_2', 'Rfa_group_3', 'Rfa_group_4', 'Date', 'Transportation Expense',
                             'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education',
                             'Children', 'Pets', 'Absenteeism Time in Hours']
        
        df = df[columns_reordered]
        
        df['Date'] = pd.to_datetime(df['Date'], format = '%d/%m/%Y')
        
        months_list = []
        for i in range(df.shape[0]):
            months_list.append(df['Date'][i].month)
            
        df['Month'] = months_list
        
        df['Day of the Week'] = df['Date'].dt.weekday
        
        df = df.drop(['Date'], axis = 1)
        
        new_col_order = ['Rfa_group_1', 'Rfa_group_2', 'Rfa_group_3', 'Rfa_group_4', 'Month', 'Day of the Week',
                         'Transportation Expense', 'Distance to Work', 'Age', 'Daily Work Load Average', 
                         'Body Mass Index', 'Education', 'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[new_col_order]
        
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})
        
        df = df.fillna(value=0)
        
        df = df.drop(['Absenteeism Time in Hours'], axis = 1)
        
        df = df.drop(['Day of the Week','Daily Work Load Average','Distance to Work'],axis=1)
        
        self.preprocessed_data = df.copy()
        
        self.data = self.scaler.transform(df)
        
    
    def predicted_probability(self):
        if (self.data is not None):
            pred = self.model.predict_proba(self.data)[:,1]
            return pred
        
    def predicted_output_category(self):
        if (self.data is not None):
            pred_outputs = self.model.predict(self.data)
            return pred_outputs
        
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.model.predict_proba(self.data)[:,1]
            self.preprocessed_data['Prediction'] = self.model.predict(self.data)
            return self.preprocessed_data


# In[ ]:





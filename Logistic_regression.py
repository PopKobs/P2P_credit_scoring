#!/usr/bin/env python
# coding: utf-8

# In[675]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[676]:


df = pd.read_csv("prosperLoanData.csv")


# In[677]:


df.head()


# In[678]:


df.columns.tolist()


# In[679]:


df.describe()


# In[680]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df.head()


# In[681]:


df["CreditScore"] = df["CreditScoreRangeLower"] + df["CreditScoreRangeUpper"] / 2


# In[682]:


#split dataset in features and target variable
#keep columns identified from EDA

#Feature variables
feature_cols = ['EmploymentStatus', 'OpenRevolvingAccounts', 'LoanOriginalAmount', 'DebtToIncomeRatio', 'BorrowerRate', 'AvailableBankcardCredit', 'ProsperRating (numeric)']
X = df[feature_cols]
#Target Variable: Credit score
y = df[['CreditScore']]


# In[683]:


X.head()


# In[684]:


df.shape


# In[685]:


# train_data = pd.concat([X, y], axis=1)
# train_data.head()
# train_data_cleaned = train_data.dropna()
# train_data_cleaned.head()'
# X_train_cleaned = train_data_cleaned[1:8]
# y_train_cleaned = train_data_cleaned[8:8]
# X_train_cleaned.head()


# In[691]:


from sklearn import preprocessing
from sklearn import utils
from sklearn.preprocessing import LabelEncoder

#convert EmploymentStatus values (String) to categorical values
#using label encoder
le = LabelEncoder()


# In[692]:


label = le.fit_transform(df['EmploymentStatus'])


# In[693]:


label


# In[694]:


df['EmploymentStatus'] = label


# In[695]:


#remove original EmploymentStatus column containing String values 
df.drop('EmploymentStatus', axis=1, inplace=True)


# In[696]:


#replace with valeus that have been converted to numbers
df.insert(0, 'EmploymentStatus', label)


# In[697]:


df.head()


# In[698]:


#reassign feature columns after 'EmploymentStatus' has been converted to categorical values
feature_cols = ['EmploymentStatus', 'OpenRevolvingAccounts', 'LoanOriginalAmount', 'DebtToIncomeRatio', 'BorrowerRate', 'AvailableBankcardCredit', 'ProsperRating (numeric)']
X = df[feature_cols]


# In[699]:


X.head()


# In[700]:


from sklearn.impute import SimpleImputer

#deal with NaN values by filling them with 0
#need to check, not sure if this is the best option? other options: forward fill, backward fill 
imputer = SimpleImputer(strategy='constant', fill_value=0)
X_imputed = imputer.fit_transform(X)


# In[701]:


# Convert the NumPy array back to a DataFrame
X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)
X = X_imputed_df


# In[702]:


na_columns = X.columns[X.isna().any()].tolist()


# In[703]:


#check, make sure there are no na columns
na_columns


# In[704]:


#create an instance of logistic regression
logreg = LogisticRegression(random_state=42)


# In[705]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.25, random_state=16)


# In[706]:


logreg.fit(X_train, y_train)
# from sklearn.linear_model import LinearRegression

# # Assuming X_train_cleaned and y_train_cleaned are your cleaned training data
# model = LinearRegression()
# model.fit(X_train, y_train)


# In[707]:


y_pred = logreg.predict(X_test)


# In[708]:


from sklearn import metrics

#confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[ ]:





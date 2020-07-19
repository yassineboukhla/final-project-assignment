import pandas as pd # importing necessary library
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder # This code will substitute in our dataset the encoded values
from sklearn import model_selection # starting to build our model: Training Phase
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sb
import pickle

data_read = pd.read_csv('original_dataset.csv') # reading our dataset and stocking it


# data_read.head(5) #showing the 5 first rows


# data_read.dtypes #showing each column's type

data_read.fillna(data_read.mean(), inplace=True) # filling empty boxes with mean value ( valeur moyenne)


# data_read.head(30)

data_read['current_loan_amount'] = data_read['current_loan_amount'].astype('int64') # converting float to int value

data_read['credit_score'] = data_read['credit_score'].astype('int64') # converting float to int value

data_read['had_bankruptcy'] = data_read['had_bankruptcy'].astype('int64') # converting float to int value

data_read['annual_income'] = data_read['annual_income'].astype('int64') # converting float to int value

# data_read.head(30) #Showing our new dataset with no empty boxes after filling them with mean values

encode_cols = data_read.iloc[:, [2, 6, 8, 10]] # extracting categorical columns

# encode_cols.head() #showing extracted categorical columns


labelencoder_encode_cols = LabelEncoder() # creating our encoder

encode_cols.iloc[:, 0] = labelencoder_encode_cols.fit_transform(encode_cols.iloc[:, 0]) # encoding the fisrt column: loan_term

encode_cols.iloc[:, 1] = labelencoder_encode_cols.fit_transform(encode_cols.iloc[:, 1]) # encoding the 2nd column: loan_industry

encode_cols.iloc[:, 2] = labelencoder_encode_cols.fit_transform(encode_cols.iloc[:, 2]) # encoding the 3rd column: business_area

encode_cols.iloc[:, 3] = labelencoder_encode_cols.fit_transform(encode_cols.iloc[:, 3]) # encoding the 4th column: property

encode_cols.head() # Showing the result of encoding

var_mod = ['loan_term', 'loan_industry', 'business_area', 'property']
le = LabelEncoder()
for i in var_mod:
    data_read[i] = le.fit_transform(data_read[i])
    
    
# data_read.head() #showing the original dataset with encoded values

array = data_read.values # starting to extract necessary columns for building our model

X = array[:, [1, 2, 3, 4 , 5, 7, 9]] # Extracting Independent columns


Y = array[:, 11] # Extracting dependent column

Y=Y.astype('int')

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.25) # 75% of data is for training, 25% test


model = DecisionTreeClassifier()
model.fit(x_train,y_train)


pickle.dump(model,open("model.pkl", "wb"))
































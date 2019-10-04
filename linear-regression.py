import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split

# reading data and handling unknowns
df = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv', na_values={
        'Year of Record': ['#N/A'],
        'Gender': [0, '#N/A', 'unknown'],
        'Age': ['#N/A'],
        'Country': [],
        'Size of City': [],
        'Profession': ['#N/A'],
        'University Degree': [0, '#N/A'],
        'Wears Glasses': [],
        'Hair Color': [0, '#N/A', 'Unknown'],
        'Body Height [cm]': [],
        'Income in EUR': []
})

# handling NaN
# try different methods of interpolate to reduce RMSE
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html#pandas.DataFrame.interpolate
df['Year of Record'] = np.floor(df['Year of Record'].interpolate(method='slinear'))
df['Gender'].fillna('other', inplace=True)
df['Age'] = np.floor(df['Age'].interpolate(method='slinear'))
# different methods of fillna?
df['Profession'].fillna(method='ffill', inplace=True)
df['University Degree'].fillna(method='ffill', inplace=True)
df['Hair Color'].fillna(method='ffill', inplace=True)
df['Income in EUR'] = df['Income in EUR'].abs()

Y = df['Income in EUR']
# features being considered for linear regression
df = df[['Year of Record', 'Gender', 'Age', 'University Degree', 'Wears Glasses', 'Hair Color', 'Body Height [cm]']]
# features not considered for linear regression "FOR NOW"
# Country, Size of City, Profession
# need to reconsider using features like 'Hair Color', 'Wears Glasses' and 'Body Height [cm]',
# these might be totally unnecessary features in predicting the income.

# Feature modifications

# Standard Scaling
yor_scaler = pp.StandardScaler()
df['Year of Record'] = yor_scaler.fit_transform(df['Year of Record'].values.reshape(-1, 1))

# One Hot Encoding
ohe_gender = pp.OneHotEncoder(categories='auto', sparse=False)
ohe_gender_data = ohe_gender.fit_transform(df['Gender'].values.reshape(len(df['Gender']), 1))
dfOneHotGender = pd.DataFrame(ohe_gender_data, columns=['Gender: '+str(i.strip('x0123_')) for i in ohe_gender.get_feature_names()])
df = pd.concat([df, dfOneHotGender], axis=1)
del df['Gender']

ohe_degree = pp.OneHotEncoder(categories='auto', sparse=False)
ohe_degree_data = ohe_degree.fit_transform(df['University Degree'].values.reshape(len(df['University Degree']), 1))
dfOneHot = pd.DataFrame(ohe_degree_data, columns=['University Degree: '+str(i.strip('x0123_')) for i in ohe_degree.get_feature_names()])
df = pd.concat([df, dfOneHot], axis=1)
del df['University Degree']

ohe_hair = pp.OneHotEncoder(categories='auto', sparse=False)
ohe_hair_data = ohe_hair.fit_transform(df['Hair Color'].values.reshape(len(df['Hair Color']), 1))
dfOneHotHair = pd.DataFrame(ohe_hair_data, columns=['Hair Color: '+str(i.strip('x0123_')) for i in ohe_hair.get_feature_names()])
df = pd.concat([df, dfOneHotHair], axis=1)
del df['Hair Color']

X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size=0.2, random_state=0)

model = linear_model.LinearRegression()
model.fit(X_train, Y_train);
y_pred = model.predict(X_test)

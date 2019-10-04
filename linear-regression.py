import pandas as pd
import numpy as np
from sklearn import preprocessing as pp

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

print(df.head())

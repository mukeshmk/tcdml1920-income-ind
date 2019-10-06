import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Check-out TF-IDF, word2vec, countvectorizer, OneHotEncoder, labelBinarizer, LabelEncoder, OrdinalEncoder

# reading data and handling unknowns
def openAndHandleUnknowns(fileName):
    return pd.read_csv(fileName, na_values={
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
def dfFillNaN(df):
    df['Year of Record'] = np.floor(df['Year of Record'].interpolate(method='slinear'))
    df['Gender'].fillna('other', inplace=True)
    df['Age'] = np.floor(df['Age'].interpolate(method='slinear'))
    # different methods of fillna?
    df['Profession'].fillna(method='ffill', inplace=True)
    df['University Degree'].fillna(method='ffill', inplace=True)
    df['Hair Color'].fillna(method='ffill', inplace=True)
    return df


# One Hot Encoding
def oheFeature(feature, encoder, data, df):
    ohedf = pd.DataFrame(data, columns=[feature + ': ' + str(i.strip('x0123_')) for i in encoder.get_feature_names()])
    df = pd.concat([df, ohedf], axis=1)
    del df[feature]
    return df


df = openAndHandleUnknowns('tcd ml 2019-20 income prediction training (with labels).csv')

df = dfFillNaN(df)
df['Income in EUR'] = df['Income in EUR'].abs()

y = df['Income in EUR']
# features being considered for linear regression
df = df[['Year of Record', 'Gender', 'Age', 'University Degree', 'Wears Glasses', 'Hair Color', 'Body Height [cm]']]
# features not considered for linear regression "FOR NOW"
# Country, Size of City, Profession
# need to reconsider using features like 'Hair Color', 'Wears Glasses' and 'Body Height [cm]',
# these might be totally unnecessary features in predicting the income.

# Feature modifications

# Standard Scaling
yor_scalar = pp.StandardScaler()
df['Year of Record'] = yor_scalar.fit_transform(df['Year of Record'].values.reshape(-1, 1))


ohe_gender = pp.OneHotEncoder(categories='auto', sparse=False)
ohe_gender_data = ohe_gender.fit_transform(df['Gender'].values.reshape(len(df['Gender']), 1))
df = oheFeature('Gender', ohe_gender, ohe_gender_data, df)

ohe_degree = pp.OneHotEncoder(categories='auto', sparse=False)
ohe_degree_data = ohe_degree.fit_transform(df['University Degree'].values.reshape(len(df['University Degree']), 1))
df = oheFeature('University Degree', ohe_degree, ohe_degree_data, df)

ohe_hair = pp.OneHotEncoder(categories='auto', sparse=False)
ohe_hair_data = ohe_hair.fit_transform(df['Hair Color'].values.reshape(len(df['Hair Color']), 1))
df = oheFeature('Hair Color', ohe_hair, ohe_hair_data, df)

# can be modified to used k-fold cross validation
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Coefficients: \n', model.coef_)
print("RMSE: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

##################################################################################################################
# Reading unlabeled instances!
sub_df = openAndHandleUnknowns('tcd ml 2019-20 income prediction test (without labels).csv')
sub_df = dfFillNaN(sub_df)
instance = pd.DataFrame(sub_df['Instance'], columns=['Instance'])
sub_df = sub_df[['Year of Record', 'Gender', 'Age', 'University Degree', 'Wears Glasses', 'Hair Color', 'Body Height [cm]']]

sub_df['Year of Record'] = yor_scalar.fit_transform(sub_df['Year of Record'].values.reshape(-1, 1))

ohe_gender_data = ohe_gender.transform(sub_df['Gender'].values.reshape(len(sub_df['Gender']), 1))
sub_df = oheFeature('Gender', ohe_gender, ohe_gender_data, sub_df)

ohe_degree_data = ohe_degree.fit_transform(sub_df['University Degree'].values.reshape(len(sub_df['University Degree']), 1))
sub_df = oheFeature('University Degree', ohe_degree, ohe_degree_data, sub_df)

ohe_hair_data = ohe_hair.fit_transform(sub_df['Hair Color'].values.reshape(len(sub_df['Hair Color']), 1))
sub_df = oheFeature('Hair Color', ohe_hair, ohe_hair_data, sub_df)

y_sub = model.predict(sub_df)
income = pd.DataFrame(y_sub, columns=['Income'])
ans = instance.join(income)

ans.to_csv('kaggle-output.csv', index=False)

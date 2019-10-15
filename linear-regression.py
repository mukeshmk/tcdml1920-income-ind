import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing as pp
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Check-out TF-IDF, word2vec, countvectorizer, OneHotEncoder, labelBinarizer, LabelEncoder, OrdinalEncoder

# reading data and handling unknowns
def openAndHandleUnknowns(fileName):
    return pd.read_csv(fileName, na_values={
        'Year of Record': [0, '#N/A', 'unknown'],
        'Gender': [0, '#N/A', 'unknown'],
        'Age': [0, '#N/A', 'unknown'],
        'Country': ['#N/A', 'unknown'],
        'Size of City': ['#N/A', 'unknown'],
        'Profession': ['#N/A'],
        'University Degree': [0, '#N/A', 'unknown'],
        'Wears Glasses': ['#N/A', 'unknown'],
        'Hair Color': [0, '#N/A', 'Unknown'],
        'Body Height [cm]': ['#N/A', 'unknown'],
        'Income in EUR': []
    })


# handling NaN
# TODO try different methods of interpolate to reduce RMSE
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


# Removing outliers using z-score
def dropNumericalOutliers(df, z_thresh=3):
    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    constrains = df.select_dtypes(include=[np.number]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh) \
        .all(axis=1)
    # Drop (inplace) values set to be rejected
    df.drop(df.index[~constrains], inplace=True)
    return df


# One Hot Encoding
def oheFeature(feature, encoder, data, df):
    # data = data[:, 1:]
    ohedf = pd.DataFrame(data, columns=[feature + ': ' + str(i.strip('x0123_')) for i in encoder.get_feature_names()])
    ohedf.drop(ohedf.columns[len(ohedf.columns) - 1], axis=1, inplace=True)
    df = pd.concat([df, ohedf], axis=1)
    del df[feature]
    return df


df = openAndHandleUnknowns('tcd ml 2019-20 income prediction training (with labels).csv')
sub_df = openAndHandleUnknowns('tcd ml 2019-20 income prediction test (without labels).csv')

df = dfFillNaN(df)
sub_df = dfFillNaN(sub_df)

df['Income in EUR'] = df['Income in EUR'].abs()
y = df['Income in EUR']
instance = pd.DataFrame(sub_df['Instance'], columns=['Instance'])

# features being considered for linear regression
features = ['Year of Record', 'Gender', 'Age', 'University Degree', 'Wears Glasses', 'Hair Color',
            'Body Height [cm]', 'Country', 'Size of City', 'Profession']

df = df[features + ['Income in EUR']]
sub_df = sub_df[features]
# need to reconsider using features like 'Hair Color', 'Wears Glasses' and 'Body Height [cm]',
# these might be totally unnecessary features in predicting the income.

# Feature modifications

# Standard Scaling
yor_scalar = pp.StandardScaler()
df['Year of Record'] = yor_scalar.fit_transform(df['Year of Record'].values.reshape(-1, 1))

sub_df['Year of Record'] = yor_scalar.transform(sub_df['Year of Record'].values.reshape(-1, 1))

age_scalar = pp.StandardScaler()
df['Age'] = age_scalar.fit_transform(df['Age'].values.reshape(-1, 1))

sub_df['Age'] = age_scalar.transform(sub_df['Age'].values.reshape(-1, 1))

# One Hot Encoding
ohe_gender = pp.OneHotEncoder(categories='auto', sparse=False)
ohe_gender_data = ohe_gender.fit_transform(df['Gender'].values.reshape(len(df['Gender']), 1))
df = oheFeature('Gender', ohe_gender, ohe_gender_data, df)

ohe_gender_data = ohe_gender.transform(sub_df['Gender'].values.reshape(len(sub_df['Gender']), 1))
sub_df = oheFeature('Gender', ohe_gender, ohe_gender_data, sub_df)

ohe_degree = pp.OneHotEncoder(categories='auto', sparse=False)
ohe_degree_data = ohe_degree.fit_transform(df['University Degree'].values.reshape(len(df['University Degree']), 1))
df = oheFeature('University Degree', ohe_degree, ohe_degree_data, df)

ohe_degree_data = ohe_degree.transform(sub_df['University Degree'].values.reshape(len(sub_df['University Degree']), 1))
sub_df = oheFeature('University Degree', ohe_degree, ohe_degree_data, sub_df)

ohe_hair = pp.OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore')
ohe_hair_data = ohe_hair.fit_transform(df['Hair Color'].values.reshape(len(df['Hair Color']), 1))
df = oheFeature('Hair Color', ohe_hair, ohe_hair_data, df)

ohe_hair_data = ohe_hair.transform(sub_df['Hair Color'].values.reshape(len(sub_df['Hair Color']), 1))
sub_df = oheFeature('Hair Color', ohe_hair, ohe_hair_data, sub_df)

# replacing the a small number of least count group values to a common feature 'other'
countryList = df['Country'].unique()
countryReplaced = df.groupby('Country').count()
countryReplaced = countryReplaced[countryReplaced['Age'] < 3].index
df['Country'].replace(countryReplaced, 'other', inplace=True)

label_country = pp.LabelEncoder()
label_country_data = label_country.fit_transform(df['Country'])
df['Country'] = pd.DataFrame(label_country_data, columns=['Country'])

# Handling the 'other' encoding in Country Feature
testCountryList = sub_df['Country'].unique()
encodedCountries = list(set(countryList) - set(countryReplaced))
testCountryReplace = list(set(testCountryList) - set(encodedCountries))
sub_df['Country'] = sub_df['Country'].replace(testCountryReplace, 'other')

label_country_data = label_country.transform(sub_df['Country'])
sub_df['Country'] = pd.DataFrame(label_country_data, columns=['Country'])

# replacing the a small number of least count group values to a common feature 'other profession'
professionList = df['Profession'].unique()
professionReplaced = df.groupby('Profession').count()
professionReplaced = professionReplaced[professionReplaced['Age'] < 3].index
df['Profession'].replace(professionReplaced, 'other profession', inplace=True)

label_prof = pp.LabelEncoder()
label_prof_data = label_prof.fit_transform(df['Profession'])
df['Profession'] = pd.DataFrame(label_prof_data, columns=['Profession'])

# Handling the 'other profession' encoding in Profession Feature
testProfessionList = sub_df['Profession'].unique()
encodedProfession = list(set(professionList) - set(professionReplaced))
testProfessionReplace = list(set(testProfessionList) - set(encodedProfession))
sub_df['Profession'] = sub_df['Profession'].replace(testProfessionReplace, 'other profession')

label_country_data = label_prof.transform(sub_df['Profession'])
sub_df['Profession'] = pd.DataFrame(label_country_data, columns=['Profession'])

del df['Income in EUR']
# can be modified to used k-fold cross validation
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)

model = rfr(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("RMSE: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

##################################################################################################################
y_sub = model.predict(sub_df)
income = pd.DataFrame(y_sub, columns=['Income'])
ans = instance.join(income)

ans.to_csv('kaggle-output.csv', index=False)

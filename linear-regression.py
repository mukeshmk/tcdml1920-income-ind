import pandas as pd

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

print(df.head())

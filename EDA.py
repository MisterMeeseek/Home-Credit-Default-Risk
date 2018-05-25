import pandas as pd
import numpy as np
import seaborn as sns

application_train = pd.read_csv('application_train.csv')

sns.countplot(x = 'AMT_INCOME_TOTAL', data = application_train)
application_train['AMT_INCOME_TOTAL'].describe().astype(int)

application_train['AMT_INCOME_TOTAL'].head(10)

pd.cut(application_train['AMT_INCOME_TOTAL'], bins = [0, 198000, 297000, 396000, 495000], include_lowest = True)

np.nanpercentile(application_train['AMT_INCOME_TOTAL'], 20, axis = 0)

sns.barplot(x = 'AMT_INCOME_TOTAL', y = 'TARGET', data = application_train)

def grouped_incomes(x):
    if x >= 200000:
        return '$200,000+'
    elif 150000 < x < 200000:
        return '$150k - $199k'
    elif 100000 < x < 150000:
        return '$100k - $149k'
    elif 50000 < x < 100000:
        return '$50k - $100k'
    elif 0 < x < 50000:
        return '$0 - $50k'
    
application_train['G_INCOMES'] = application_train['AMT_INCOME_TOTAL'].apply(grouped_incomes)

sns.countplot(x = application_train['G_INCOMES'].sort_values())
sns.barplot(x = 'G_INCOMES', y = 'TARGET', data = application_train)

def grouped_credit(x):
    if x >= 800000:
        return 'Extremely high limit'
    elif 500000 < x < 800000:
        return 'Very high limit'
    elif 270000 < x < 500000:
        return 'High limit'
    elif 45000 < x < 270000:
        return 'Moderate limit'
    elif x <= 45000:
        return 'Low limit'

application_train['G_CREDIT'] = application_train['AMT_CREDIT'].apply(grouped_credit)
sns.countplot(x = 'G_CREDIT', data = application_train, orient = 'v')
sns.distplot(application_train['AMT_CREDIT'])


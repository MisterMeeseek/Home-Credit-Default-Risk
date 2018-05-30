import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter

application_train = pd.read_csv('application_train.csv')

# Create null values where 'XNA' and 'XAP' are used, function that takes a series as an argument
def replace_with_null(x):
    if x == 'XNA':
        return np.nan
    elif x == 'XNP':
        return np.nan
    else:
        return x
    
'''# Detect outliers with Tukey method  <<<< this is either fucking up or isn't the best approach to dealing with outliers in this dataset. Failed to detect any outliers in the income data even though a number of them exist
def detect_outliers(df, n, features):
    outlier_indices = []
    for col in features:
        Q1 = np.nanpercentile(df[col], 25)
        Q3 = np.nanpercentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
        outlier_indices = Counter(outlier_indices)
        multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
        return multiple_outliers'''

# Target variable distribution check
sns.countplot('TARGET', data = application_train)

# Type of loan
sns.countplot('NAME_CONTRACT_TYPE', data = application_train)
sns.barplot(x = 'NAME_CONTRACT_TYPE', y = 'TARGET', hue = 'CODE_GENDER', data = application_train)

# Gender
application_train['CODE_GENDER'] = application_train['CODE_GENDER'].apply(replace_with_null) # transform XNA's to null
sns.countplot(x = 'CODE_GENDER', data = application_train)
gender_to_default = sns.barplot(x = 'CODE_GENDER', y = 'TARGET', data = application_train) # men appear to be noticeably more likely to have payment difficulties even though they make up the minority of applicants

# Car Ownership
car_dist = sns.countplot('FLAG_OWN_CAR', data = application_train, hue = 'CODE_GENDER')
sns.barplot(x = 'FLAG_OWN_CAR', y = 'TARGET', data = application_train, hue = None)
sns.barplot(x = 'FLAG_OWN_CAR', y = 'TARGET', data = application_train) # notable difference between men who do(n't) own a car and payment difficulty probability

# Realty Ownership
sns.countplot('FLAG_OWN_REALTY', data = application_train, hue = 'CODE_GENDER')
sns.barplot(x = 'FLAG_OWN_REALTY', y = 'TARGET', data = application_train) # no significant difference between realty owners and default probability
sns.barplot(x = 'FLAG_OWN_REALTY', y = 'TARGET', data = application_train, hue = 'CODE_GENDER') # follows same gender trends in line with sample distribution and higher probability correlated with male applicants

# Number of children
sns.distplot(a = application_train['CNT_CHILDREN'], kde = True) # some extreme outliers exist here (15+ kids!?)

def group_kids(x):
    if x >= 10:
        return '10 or more'
    else:
        return x

application_train['Children_Qty'] = application_train['CNT_CHILDREN'].apply(group_kids) # to group applicants with 10 or more kids
sns.countplot(x = 'Children_Qty', data = application_train) # vast majority of applicants don't have any children
sns.barplot(x = 'Children_Qty', y = 'TARGET', data = application_train) # probability of payment difficulties spikes at 6 or more kids

# Total annual income of borrowers
application_train['AMT_INCOME_TOTAL'].describe().astype(int)
highest_incomes = application_train['AMT_INCOME_TOTAL'].loc[(application_train['AMT_INCOME_TOTAL'] >= 202500) & (application_train['AMT_INCOME_TOTAL'] <= 1000000)]

    # detecting outliers in the applicants' income data
income_outliers = detect_outliers(application_train, 1, ['AMT_INCOME_TOTAL'])
len(application_train['AMT_INCOME_TOTAL'])

    # getting better view of distribution of incomes
len(application_train.loc[(application_train['AMT_INCOME_TOTAL'] >= 202500) & (application_train['AMT_INCOME_TOTAL'] <= 1000000)])
len(application_train.loc[application_train['AMT_INCOME_TOTAL'] > 1000000]) # millionaires are relatively rare in this sample, seems intuitive but still not sure how these people fall into the "unbanked" population

incomes_below_500k = application_train['AMT_INCOME_TOTAL'].loc[application_train['AMT_INCOME_TOTAL'] <= 500000]

sns.distplot(a = highest_incomes, kde = False) # marginally skewed right, very steep center (at least with log transformed values)
sns.distplot(a = incomes_below_500k, kde = True) # still right skewed after treating for outliers above $500k
sns.barplot(x = incomes_below_500k, y = application_train['TARGET'])

    # binning incomes into groups
def grouped_incomes(x):
    if x >= 500000:
        return '$500k+'
    elif 450000 <= x < 500000:
        return '$450k - $499k'
    elif 400000 <= x < 450000:
        return '$400k - $449k'
    elif 350000 <= x < 400000:
        return '$350k - $399k'
    elif 300000 <= x < 350000:
        return '$300k - $349k'
    elif 250000 <= x < 300000:
        return '$250k - $299k'
    elif 200000 <= x < 250000:
        return '$200k - $249k'
    elif 150000 <= x < 200000:
        return '$150k - $199k'
    elif 100000 <= x < 150000:
        return '$100k - $149k'
    elif 50000 <= x < 100000:
        return '$50k - $100k'
    elif 0 <= x < 50000:
        return '$0 - $50k'
    
application_train['G_INCOMES'] = application_train['AMT_INCOME_TOTAL'].apply(grouped_incomes)

grouped_incomes_cntplt = sns.countplot(x = application_train['G_INCOMES'])
plt.setp(grouped_incomes_cntplt.get_xticklabels(), rotation = 45)

incomes_grouped = application_train['G_INCOMES'].value_counts()
application_train['G_INCOMES'].describe()
sns.distplot(application_train['G_INCOMES'])
sns.barplot(x = incomes_grouped, y = 'TARGET', data = application_train)
application_train['G_INCOMES']

# Total Amount of Credit
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

sns.distplot(application_train['AMT_INCOME_TOTAL'].loc[application_train['AMT_INCOME_TOTAL'] > 5000000])


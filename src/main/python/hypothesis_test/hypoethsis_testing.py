import pandas as pd
from scipy import stats

df = pd.DataFrame()

# Performing a paired t-test (that is, difference between two columns since they are colinear)
test_stat, p_value = stats.ttest_rel(df['New Scheme'], df['Old Scheme'], alternative='greater')

print('The p-value is', p_value)

if p_value < 0.05:
    print(f'As the p-value {p_value} is less than the level of significance, we reject the null hypothesis.')
else:
    print(f'As the p-value {p_value} is greater than the level of significance, we fail to reject the null hypothesis.')

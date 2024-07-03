# Note: may have to remove last line in combined_data.csv before running this...
# It may be containing empty values


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load
data = pd.read_csv('combined_data.csv', parse_dates=['DATE'])


# Pairplot
# sns.pairplot(data)
# plt.show()

# Correlation matrix
# corr_matrix = data.corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# plt.show()

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import numpy as np
import statsmodels.api as sm

# Define the features and target variable
X = data[['HOUST', 'M2SL', 'PPIACO', 'UNRATE']]
y = data['VIX']

# Define the model with a range of alpha values for regularization
alphas = np.logspace(-6, 6, 13)
model = RidgeCV(alphas=alphas, cv=10, scoring='neg_mean_squared_error')

# Fit
model.fit(X, y)

# cross-validation
mse_scorer = make_scorer(mean_squared_error)
r2_scorer = make_scorer(r2_score)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

mse_scores = cross_val_score(model, X, y, scoring=mse_scorer, cv=kf)
r2_scores = cross_val_score(model, X, y, scoring=r2_scorer, cv=kf)

# Print results
print(f'Mean Squared Error: {np.mean(mse_scores)} ± {np.std(mse_scores)}')
print(f'R-squared: {np.mean(r2_scores)} ± {np.std(r2_scores)}')

# coefficients
coef = pd.Series(model.coef_, index=X.columns)
print(coef)

# Add an intercept
X_with_const = sm.add_constant(X)

# Fit the OLS regression
ols_model = sm.OLS(y, X_with_const).fit()

# Get p-values
params = ols_model.params
p_values = ols_model.pvalues

print('Coefficients:')
print(params)
print('P-values:')
print(p_values)


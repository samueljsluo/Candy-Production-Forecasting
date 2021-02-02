import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def create_time_series_features(df, label):
    df['date'] = df.index
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['weekofyear'] = df['date'].dt.weekofyear

    df['lag_month'] = df['amount'].shift(1)
    df['lag_quarter'] = df['amount'].shift(4)
    df['lag_year'] = df['amount'].shift(12)

    df['rolling_mean_month'] = df['amount'].rolling(window=1).mean()
    df['rolling_sum_month'] = df['amount'].rolling(window=1).sum()
    df['rolling_max_month'] = df['amount'].rolling(window=1).max()
    df['rolling_min_month'] = df['amount'].rolling(window=1).min()
    df['rolling_std_month'] = df['amount'].rolling(window=1).std()

    df['rolling_mean_quarter'] = df['amount'].rolling(window=4).mean()
    df['rolling_sum_quarter'] = df['amount'].rolling(window=4).sum()
    df['rolling_max_quarter'] = df['amount'].rolling(window=4).max()
    df['rolling_min_quarter'] = df['amount'].rolling(window=4).min()
    df['rolling_std_quarter'] = df['amount'].rolling(window=4).std()

    X = df.loc[:, (df.columns != label) & (df.columns != 'date')]
    return X, df[label]


df = pd.read_csv('data/candy_production.csv', parse_dates=['observation_date'])
df = df.rename(columns={'observation_date': 'date', 'IPG3113N':'amount'})
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')
df = df.set_index('date')

production_train = df.loc[df.index < '2008-12-01'].copy()
production_test = df.loc[df.index >= '2008-12-01'].copy()

X_train, y_train = create_time_series_features(production_train, label='amount')
X_test, y_test = create_time_series_features(production_test, label='amount')


model = xgb.XGBRegressor()

model.fit(X_train, y_train)

production_test['amount_prediction'] = model.predict(X_test)
production_all = pd.concat([production_test, production_train], sort=False)

graph = production_all[['amount', 'amount_prediction']].plot(figsize=(15, 10))
plt.title('Forecasting of Candy Production')
plt.show()

print('Mean Squared Error:{}'.format(mean_squared_error(production_test['amount'],
                                                        production_test['amount_prediction'])))
print('Mean Absolute Error:{}'.format(mean_absolute_error(production_test['amount'],
                                                              production_test['amount_prediction'])))


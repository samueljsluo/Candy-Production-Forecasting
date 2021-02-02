# US Candy Production Forecasting

### This project used XGBoost to predict the candy production.

### The data can be download at https://www.kaggle.com/rtatman/us-candy-production-by-month

## The distribution of production
![image](https://github.com/samueljsluo/CandyProductionForecasting/blob/main/data/Production%20Distribution.png)

## Outlier Detection
![image](https://github.com/samueljsluo/CandyProductionForecasting/blob/main/data/Outlier%20Detection.png)

## Lag analysis and autocorrelation analysis

### Lag 1
![image](https://github.com/samueljsluo/CandyProductionForecasting/blob/main/data/lag_1.PNG)

### Lag 4
![image](https://github.com/samueljsluo/CandyProductionForecasting/blob/main/data/lag_4.PNG)

### Lag 12
![image](https://github.com/samueljsluo/CandyProductionForecasting/blob/main/data/lag_12.PNG)

Results:
| Metrix              | value              |
|---------------------|--------------------|
| Mean Squared Error  | 0.3239758131874296 |
| Mean Absolute Error | 0.4282815776134681 |

![image](https://github.com/samueljsluo/CandyProductionForecasting/blob/main/data/forcasting.png)
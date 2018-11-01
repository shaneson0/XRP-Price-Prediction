

from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
series = read_csv(r'\xrp.csv',header=0,parse_dates=[0],index_col=0,squeeze=True)
series = series[::-1]
series = series['Close']
X = series.values
train, test = X[0:700], X[700:len(X)]
history = [x for x in train]
predictions = list()
for key in test:
    #data = np.array(series)
    result = None
    arima = ARIMA(test, order=(2,1,2))
    result = arima.fit(disp=False)
    pred = result.predict(1,67,typ='levels')
    x = pd.date_range('2018-08-25','2018-10-30')
print(pred)
sum = 0
for i in range(60):
   sum += (pred[i]-test[i])**2
r =(sum/60)**0.5
print(r)
plt.figure(figsize=(10,5))
plt.plot(x[:60],test[:60] ,label='Data')
plt.plot(x,pred,label='ARIMA Model')
plt.xlabel('Days')
plt.ylabel('Values')
plt.legend()
plt.show()

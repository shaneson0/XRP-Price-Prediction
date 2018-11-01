
import pandas as pd
import json


df = pd.read_csv('./data/xrp.csv', sep=',',header=0)

XRP = df[['Date', 'Close']]

print(df['Date'].values)

dic = {
    'Date': list(df['Date'].values),
    'Close': list(df['Close'].values)
}

JsonStr = json.dumps(dic)
with open('./data/data.json', 'w') as fw:
    json.dump(JsonStr, fw)






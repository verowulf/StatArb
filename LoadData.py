### 1. Gather data ###

import pandas as pd
from Settings import N

# TRI: Total Return Index
#df_TRI = pd.read_excel('../input/raw_data.xlsx', sheetname='TR', header=[0, 1], index_col=0)
#df_TRI.to_csv('../input/df_TRI.csv')
df_TRI = pd.read_csv('../input/df_TRI.csv', header=[0, 1], index_col=0)
#df_TRI.head(3)
df_TRI.info()    # need at least 1241 (= 240 + 750 + 250 + 1) rows

# CAP: market CAP
#df_CAP = pd.read_excel('../input/raw_data.xlsx', sheetname='MV', header=[0, 1], index_col=0)
#df_CAP.to_csv('../input/df_CAP.csv')
#df_CAP = pd.read_csv('../input/df_CAP.csv', header=[0, 1], index_col=0)
#df_CAP.head(3)
#df_CAP.info()


### 2. Data preprocessing ###

#from pandas_datareader.data import DataReader
#import datetime
#start_date = datetime.date(1989,  1,  5)
#end_date   = datetime.date(2016, 10, 13)

#(DataReader('^GSPC', 'yahoo', start_date, end_date)).to_csv('../input/SPX.csv')
df_SPX = pd.read_csv('../input/SPX.csv', index_col=0, parse_dates=True, na_values=['nan'])
df_SPX.index.name = None
df_SPX.columns = pd.MultiIndex(levels=[range(6), range(6)], labels=[range(6), range(6)], names=['Name', 'Code'])
df_TRI = (df_SPX.join(df_TRI, how='left')).iloc[:, 6:]
#df_CAP = (df_SPX.join(df_CAP, how='left')).iloc[:, 6:]
#df_TRI.tail(3)

'''
m = df_CAP.shape[0]
n = df_CAP.shape[1]
for j in range(n):
    for i in reversed(range(m)):
        if df_CAP.iloc[i, j] == df_CAP.iloc[i-1, j]:
            df_CAP.iloc[i, j] = np.nan
        else:
            break

df_CAP.to_csv('../input/df_CAP_NaN.csv')
'''

df_CAP_NaN = pd.read_csv('../input/df_CAP_NaN.csv', header=[0, 1], index_col=0)
df_CAP_NaN.info()

df_CAP_f = df_CAP_NaN.fillna(-1)    # fill NaN with -1 because Python sorting ranks them at the top
Argsort = ((df_CAP_f.values).argsort())[:, ::-1]
Argsort_N = Argsort[:, :N]    # N largest market CAPs
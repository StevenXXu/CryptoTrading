# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 22:37:37 2018

@author: SX
"""

import json
import numpy as np
import os
import pandas as pd
import urllib.request
 
def JSONDictToDF(d):
    '''
    Converts a dictionary created from json.loads to a pandas dataframe
    d:      The dictionary
    '''
    n = len(d)
    cols = []
    if n > 0:   #Place the column in sorted order
        cols = sorted(list(d[0].keys()))
    df = pd.DataFrame(columns = cols, index = range(n))
    for i in range(n):
        for coli in cols:
            df.set_value(i, coli, d[i][coli])
    return df
     
def GetAPIUrl(cur):
    '''
    Makes a URL for querying historical prices of a cyrpto from Poloniex
    cur:    3 letter abbreviation for cryptocurrency (BTC, LTC, etc)
    '''
    u = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_' + cur + '&start=1420070400&end=9999999999&period=14400'
    return u
 
def GetCurDF(cur, fp):
    '''
    cur:    3 letter abbreviation for cryptocurrency (BTC, LTC, etc)
    fp:     File path (to save price data to CSV)
    '''
    openUrl = urllib.request.urlopen(GetAPIUrl(cur))
    r = openUrl.read()
    openUrl.close()
    d = json.loads(r.decode())
    df = JSONDictToDF(d)
    df.to_csv(fp, sep = ',')
    return df
 
#%%Path to store cached currency data
datPath = 'CurDat/'
if not os.path.exists(datPath):
    os.mkdir(datPath)
#Different cryptocurrency types
cl = ['BTC', 'LTC', 'ETH', 'DASH']
#Columns of price data to use
CN = ['close', 'high', 'low', 'open', 'volume']
#Store data frames for each of above types
D = []
for ci in cl:
    dfp = os.path.join(datPath, ci + '.csv')
    try:
        df = pd.read_csv(dfp, sep = ',')
    except FileNotFoundError:
        df = GetCurDF(ci, dfp)
    D.append(df)
#%%Only keep range of data that is common to all currency types
cr = min(Di.shape[0] for Di in D)
for i in range(len(cl)):
    D[i] = D[i][(D[i].shape[0] - cr):]
    
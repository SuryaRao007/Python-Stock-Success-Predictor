#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:28:15 2021
o SE
1 LVGO
2 TENB
3 FTCH
4 SDC
5 PLUG
6 ADBE
7 VISA
8 NVTA
@author: surya rao
Variables from WGF stocks analysis group Agni Pariksha copyright Anil Bhojani
"""

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import plot_decision_regions as pdr
import pandas as pd
from io import StringIO

df = pd.read_csv("StocksAnalysis.csv",header=None)
df.columns =['Outcome','Disruptor','Moat','CEO','USBased','TAM','MarketShare'
             ,'Usage','PriorRevGrowth','FutureRevGrowth','EarningsGrowth'
             ,'SizeAndTime','International','Expert','TipsRank','RuleBreaker'
             ,'GrowthPS','FounderStake','InstStake','Phase','ShortInterest']
print(df['Outcome'])
Ydata =df['Outcome'].values
Xdata =df.iloc[:,1:].values
lr = LogisticRegression(penalty ='l2', C=1.0)
lr.fit(Xdata,Ydata)
print(lr.coef_)
coeffecients = pd.DataFrame(lr.coef_,columns =df.columns[1:])
coeffecients.plot(kind="barh",figsize=(20,20), legend = 'reverse',sort_columns = True)

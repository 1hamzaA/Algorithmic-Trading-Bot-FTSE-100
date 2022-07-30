import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
import pandas_datareader as pdr
import datetime
import statsmodels.api as sm
import re
from sklearn import linear_model
from statsmodels.formula.api import ols
import requests
import dateutil
import urllib

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


train_data2=[]
train_data2 = pd.DataFrame(train_data2, columns=["Date","Open","Close","High","Low","Volume","FTSE Close",
                                                 "PctChange","FTSE PctChange","PctChange Trend","FTSE PctChange Trend",
                                                 "Ticker","Upper BB","Lower BB","Point of Inflection","Value of Inflection",
                                                 "P/L%", "Transaction ID", "Length of Trade", "TP", 
                                                 "SL", "Trade Outcome"])

train_data4=[]
train_data4 = pd.DataFrame(train_data4, columns=["Date","Ticker","Buy/Hold","Buy Price (BP)",
                                                 "Sell/Hold","P/L%","Length of Trade",
                                                 "TP","SL"])
baseUrl = 'https://query1.finance.yahoo.com/v7/finance/download'


###############################################
####      Select ticker and timeframe      ####
###############################################
t1=400

for ticker in ["AAL.L","ANTO.L","AZN.L","BKG.L","BLND.L","BRBY.L","CPG.L","CRH.L","DCC.L","SMDS.L","EZJ.L","FERG.L","FRES.L","HLMA.L","HIK.L","INF.L","JD.L","JMAT.L","JET.L","LAND.L","MRO.L","MNDI.L","OCDO.L","PSON.L","RMV.L","RIO.L","RDSA.L","RMG.L","SGE.L","SGRO.L","SN.L","SMIN.L","SSON.L","SPX.L","TW.L","TSCO.L","WTB.L"]:
#for ticker in ["PHNX.L","LAND.L","LGEN.L" ]:   
    
 print(ticker)
 try:   
   
     def timestamp(dt):
         return round(datetime.datetime.timestamp(dt))
    
     def get_csv_data(ticker=ticker, days=t1) :
        endDate = datetime.datetime.today()
        startDate = endDate - datetime.timedelta(days=days)
        useragent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.82 Safari/537.36"
        response = requests.get(baseUrl+"/"+urllib.parse.quote(ticker), stream=True, headers={"USER-AGENT":useragent}, params = {
            'period1': timestamp(startDate),
            'period2': timestamp(endDate),
            'interval': '1d',
            'events': 'history',
            'includeAdjustedClose': 'true'
        })
        response.raise_for_status()
        return pd.read_csv(response.raw)
            
     x1 = get_csv_data()
     x1 = x1.set_index('Date')


                           
    
     #print(x1)
     
    # Returning index of date that we want to start creating test data from (From 2018-01-01 is Test Data) #
    
     df=x1['Close']
     df=df.reset_index()
     mask = (df['Date'] >= '2020-12-24') & (df['Date'] <= '2020-12-25')
     StartOfTestIndex=df.loc[mask].reset_index()['index']
     
     df= df.set_index('Date') 

     train_data= df
     train_data1= train_data
     train_data.dropna(inplace=True)
     
     train_data=pd.DataFrame(train_data,index = train_data.index,columns=['Close'])
     train_data1=train_data
     train_data1["Ticker"]=ticker
     train_data1["Open"]=x1["Open"]
     train_data1["High"]=x1["High"]
     train_data1["Low"]=x1["Low"]
     train_data1["Volume"]=x1["Volume"]
    
     MASpan = 20
     BBSpan = 20
     MA1Span = 5
     MA2Span = 20
     MA3Span = 30
     MA4Span = 100
     MA5Span = 150
     PIUpperBand = 0.01
     PILowerBand = 0.01  
      
      
      
    
     train_data1['MA'] = train_data1['Close'].rolling(MASpan).mean()
    
     train_data1['STD'] = train_data1['Close'].rolling(BBSpan).std()
     train_data1['Upper BB'] = train_data1['MA'] + (train_data1['STD'] * 2)
     train_data1['Lower BB'] = train_data1['MA'] - (train_data1['STD'] * 2)
     del train_data1['MA']
     del train_data1['STD']
     train_data1['BBRange']=(train_data1['Close']-train_data1['Lower BB'])/(train_data1['Upper BB']-train_data1['Lower BB'])
     train_data1['PIU']=(train_data1['Close']/train_data1['Upper BB'])-1
     train_data1['PID']=(train_data1['Close']/train_data1['Lower BB'])-1
    
    
    ###############################################
    #### Point of Inflection ~ Bollinger Bands ####
    ###############################################
    
    # Identifying points on inflection and direction of trend post inflection
    
     conditions1 = [
     (train_data1['PIU'] >= -PIUpperBand) & (train_data1['PID'] >= 0),
     (train_data1['PIU'] <= 0) & (train_data1['PID'] <= PILowerBand)
     ]
         
     values1 = ["Upper PI","Lower PI"]
      
     train_data1['Point of Inflection'] = np.select(conditions1, values1)
      
     train_data1.loc[(train_data1['Point of Inflection']!="Upper PI") & (train_data1['Point of Inflection']!="Lower PI"),'Point of Inflection']  = np.nan
    
     
     
    # Extracting price at point of inflection    
    
     conditions2 = [
               (train_data1['Point of Inflection'] != "Upper PI") & (train_data1['Point of Inflection'] != "Lower PI")
               ]
    
     values2 = [float("Nan")]
      
     train_data1['Value of Inflection'] = np.select(conditions2, values2, default= train_data1['Close'])
     train_data1['Value of Inflection'] = train_data1['Close']
     
     
     del train_data1['BBRange']
     del train_data1['PIU']
     del train_data1['PID']
       
      
     train_data1["PctChange"]= train_data1['Close'].pct_change()
    
     
    
    
    # Extracting price at point of inflection    
     conditions2 = [
               (train_data1['Point of Inflection'] != "Upper PI") & (train_data1['Point of Inflection'] != "Lower PI")]
    
     values2 = [-999]
     train_data1['Value of Inflection'] = np.select(conditions2, values2, default= train_data1['Close'])
    
    
    
     conditions4 = [
               (train_data1['Point of Inflection'] != "Upper PI")
               ]
      
     values4 = [float(1)]
     train_data1['PI Point'] = np.select(conditions4, values4, default= float(0))
                                                     
    
    
     conditions3 = [(train_data1['Point of Inflection'] == train_data1['Point of Inflection'].shift(1)) & (train_data1['Point of Inflection'] != "Upper PI")]
     values3 = [0]
     train_data1['Value of Inflection'] = np.select(conditions3, values3, default= train_data1['Close'])
                                                 
     del train_data1['PI Point']
     train_data1['Value of Inflection']=train_data1['Close']
        
     train_data1 = train_data1.reset_index()
     train_data2 = train_data2.append(train_data1)
     train_data1 = train_data2
 except:
     print("Fail")
     

print(train_data1.groupby(['Ticker'])['Volume'].count())



###############################################
####               Parameters              ####
###############################################
for n in [5]:  # Span for Oscillators
 for TakeProfit in [x*0.02 for x in range(2) if x != 0]:
  for StopLoss in [x*(-0.07) for x in range(2) if x != 0]:
      
 
    print("n="+str(n)+" _ "+"TP="+str(TakeProfit)+" _ "+"SL="+str(StopLoss))
    train_data1['Iteration'] = "n="+str(n)+" _ "+"TP="+str(TakeProfit)+" _ "+"SL="+str(StopLoss)
    
    

    ###############################################
    ####                  RSI                  ####
    ###############################################          
    def RSIfun(price, n):
        delta = price['Close'].diff()
        #-----------
        dUp, dDown = delta.copy(), delta.copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0
    
        RolUp = dUp.rolling(n).mean()
        RolDown = abs(dDown.rolling(n).mean())
       # RolUp = pd.rolling_mean(dUp, n)
        #RolDown = pd.rolling_mean(dDown, n).abs()
    
        RS = RolUp / RolDown
        rsi= 100.0 - (100.0 / (1.0 + RS))
        return rsi
    
    train_data1['RSI'] = RSIfun(train_data1, n)
    
      
    ###############################################
    ####         Calculate Oscillators         ####
    ###############################################
     
    
    train_data1['Smoothed RSI'] = train_data1['RSI'].ewm(span=n, adjust=False).mean() 
    train_data1['Double Smoothed RSI'] = train_data1['Smoothed RSI'].ewm(span=n, adjust=False).mean() 
    train_data1['Signal Line'] = train_data1['Double Smoothed RSI'].rolling(window=n).mean()
    train_data1['Derivative Oscillator'] = train_data1['Double Smoothed RSI']-train_data1['Signal Line']
     
    train_data1['Typical Price (TP)'] = (train_data1['High']+train_data1['Low']+train_data1['Close'])/3
    train_data1['TPV'] = train_data1['Typical Price (TP)']*train_data1['Volume']
    train_data1['VWAP'] = train_data1['Typical Price (TP)']
    train_data1['MVWAP'] = train_data1['TPV'].rolling(n).sum()/train_data1['Volume'].rolling(n).sum()
     
     
    train_data1['Slope'] = train_data1['MVWAP'].rolling(window=n).apply(lambda x: np.polyfit(np.array(range(0,n)), x, 1)[0], raw=True)
    train_data1['Smoothed Slope'] = train_data1['Slope'].ewm(span=n, adjust=False).mean() 
    train_data1['Double Smoothed Slope'] = train_data1['Smoothed Slope'].ewm(span=n, adjust=False).mean() 
    train_data1['Slope Signal Line'] = train_data1['Double Smoothed Slope'].rolling(window=n).mean()
    train_data1['Slope Oscillator'] = train_data1['Double Smoothed Slope']-train_data1['Slope Signal Line']
     

    del train_data1['Smoothed RSI']
    del train_data1['Double Smoothed RSI']
    del train_data1['Signal Line']
    del train_data1['Typical Price (TP)']
    del train_data1['TPV']
    del train_data1['VWAP']
    del train_data1['Smoothed Slope'] 
    del train_data1['Double Smoothed Slope'] 
    del train_data1['Slope Signal Line']
                  
     
    train_data1['Buy/Hold'] = ""
    train_data1['Buy Price (BP)'] = ""
    train_data1['Open % vs BP'] = ""
    train_data1['Close % vs BP'] = ""
    train_data1['High % vs BP'] = ""
    train_data1['Low % vs BP'] = ""
    train_data1['Sell/Hold'] = ""
    train_data1['P/L%'] = ""
    
    train_data1['Buy Price (BP)'] = pd.to_numeric(train_data1['Buy Price (BP)'], errors='coerce')
    train_data1['Open % vs BP'] = pd.to_numeric(train_data1['Open % vs BP'], errors='coerce')
    train_data1['Close % vs BP'] = pd.to_numeric(train_data1['Close % vs BP'], errors='coerce')
    train_data1['High % vs BP'] = pd.to_numeric(train_data1['High % vs BP'], errors='coerce')
    train_data1['Low % vs BP'] = pd.to_numeric(train_data1['Low % vs BP'], errors='coerce')
    train_data1['P/L%'] = pd.to_numeric(train_data1['P/L%'], errors='coerce')

    
    
    
    ###############################################
    ####              Buy Signals              ####
    ###############################################
    
    
    train_data1 = train_data1.reset_index(drop=True)
    for index in range(0, len(train_data1)-1):
     print(str(index) + " : "+ str(len(train_data1)))
     
     
    # Creating Buy signals based on below criteria
    
     try:
      conditions = [
          (train_data1['Sell/Hold'][[index-1]].str[:4]=="Hold"),
          (train_data1['Sell/Hold'][[index-1]].str[:4]=="Sell"),
          (train_data1['Ticker'][[index]] == train_data1['Ticker'][index-1]) & (train_data1['Slope Oscillator'][[index]] > train_data1['Slope Oscillator'][index-1]) & (train_data1['Slope Oscillator'][[index]] > -10) & (train_data1['Derivative Oscillator'][[index]] > 0) & (train_data1['Derivative Oscillator'][[index]] > train_data1['Derivative Oscillator'][index-1]) & (train_data1['Derivative Oscillator'][index-1] < 0) & (train_data1['Derivative Oscillator'][[index]] != "")
          ]
          
     except:
      conditions = [
          (train_data1['Sell/Hold'][[index]].str[:4]=="Hold"),
          (train_data1['Sell/Hold'][[index]].str[:4]=="Sell"),
          (train_data1['Ticker'][[index]] == train_data1['Ticker'][[index]]) & (train_data1['Slope Oscillator'][[index]] > train_data1['Slope Oscillator'][[index]]) & (train_data1['Slope Oscillator'][[index]] > -10) & (train_data1['Derivative Oscillator'][[index]] > 0) & (train_data1['Derivative Oscillator'][[index]] > train_data1['Derivative Oscillator'][[index]]) & (train_data1['Derivative Oscillator'][[index]] < 0) & (train_data1['Derivative Oscillator'][[index]] != ""),
          ]   
     values = ['Hold', "", 'Buy at Open']
     train_data1['Buy/Hold'][[index]] = np.select(conditions, values, default= "")

    
    
    
    ###############################################
    ####               Buy Price               ####
    ###############################################
    
    # Creating Buy signals based on below criteria
    
     try:
      values = [train_data1['Buy Price (BP)'][[index-1]], np.nan, train_data1['Open'][[index]]]
     except:
      values = [train_data1['Buy Price (BP)'][[index]], np.nan, train_data1['Open'][[index]]]   
     train_data1['Buy Price (BP)'][[index]] = np.select(conditions, values, default= np.nan)
     #train_data1['Buy Price (BP)'] = pd.to_numeric(train_data1['Buy Price (BP)'], errors='coerce')
    #print(train_data1)
     
     
    ###############################################
    ####     Close/High/Low % vs Buy Price     ####
    ###############################################
    
     train_data1['Open % vs BP'][[index]] = (train_data1['Open'][[index]]/train_data1['Buy Price (BP)'][[index]])-1
     train_data1['Close % vs BP'][[index]] = (train_data1['Close'][[index]]/train_data1['Buy Price (BP)'][[index]])-1
     train_data1['High % vs BP'][[index]] = (train_data1['High'][[index]]/train_data1['Buy Price (BP)'][[index]])-1
     train_data1['Low % vs BP'][[index]] = (train_data1['Low'][[index]]/train_data1['Buy Price (BP)'][[index]])-1
     
    ###############################################
    ####             Selling Point             ####
    ###############################################
    
     conditions = [
             (train_data1['Open % vs BP'][[index]] >= TakeProfit),
             (train_data1['Open % vs BP'][[index]] <= StopLoss),
             (train_data1['High % vs BP'][[index]] >= TakeProfit) & (train_data1['Low % vs BP'][[index]] > StopLoss),
             (train_data1['Low % vs BP'][[index]] <= StopLoss) & (train_data1['High % vs BP'][[index]] < TakeProfit),
             (train_data1['High % vs BP'][[index]] >= TakeProfit) & (train_data1['Low % vs BP'][[index]] <= StopLoss),
             (train_data1['High % vs BP'][[index]] < TakeProfit) & (train_data1['Low % vs BP'][[index]] > StopLoss)
             
             ]
     values = ["Sell based upon Open TP","Sell based upon Open SL","Sell based upon TP", "Sell based upon SL", "Sell neutral", "Hold"]
     train_data1['Sell/Hold'][[index]] = np.select(conditions, values)
     train_data1['Sell/Hold'] = train_data1['Sell/Hold'].astype("str")



    ###############################################
    ####                  P/L%                 ####
    ###############################################   
     conditions = [
             (train_data1['Open % vs BP'][[index]] >= TakeProfit),
             (train_data1['Open % vs BP'][[index]] <= StopLoss),
             (train_data1['High % vs BP'][[index]] >= TakeProfit) & (train_data1['Low % vs BP'][[index]] > StopLoss),
             (train_data1['Low % vs BP'][[index]] <= StopLoss) & (train_data1['High % vs BP'][[index]] < TakeProfit),
             (train_data1['High % vs BP'][[index]] >= TakeProfit) & (train_data1['Low % vs BP'][[index]] <= StopLoss), #"Sell neutral" 
             (train_data1['High % vs BP'][[index]] < TakeProfit) & (train_data1['Low % vs BP'][[index]] > StopLoss)
             ]
    
     values = [train_data1['Open % vs BP'][[index]],train_data1['Open % vs BP'][[index]],TakeProfit, StopLoss, int(0), np.nan]
     train_data1['P/L%'][[index]] = np.select(conditions, values, default = np.nan)

    
     
    ###############################################
    ####    Loop to correct action sequence    ####
    ###############################################
    
    def AdjustActions(i):
         conditions = [
             (train_data1['Sell/Hold'].shift(1) == "Hold") & (train_data1['Ticker'] == train_data1['Ticker'].shift(1))
             ]
         values = ["Hold"]
         train_data1['Buy/Hold'] = np.select(conditions, values, default= train_data1['Buy/Hold'])
         
         
         conditions = [
             (train_data1['Sell/Hold'].shift(1) == "Hold") & (train_data1['Ticker'] == train_data1['Ticker'].shift(1))
             ]
         values = [train_data1['Buy Price (BP)'].shift(1)]
         train_data1['Buy Price (BP)'] = np.select(conditions, values, default= train_data1['Buy Price (BP)'])
         return i
     
    AdjustActions(1)
     
     
    for i in range(50):
         AdjustActions(i)
                     
   
    
    ###############################################
    ####            Transaction ID             ####
    ###############################################  
 
    train_data1 = train_data1.reset_index()
    del train_data1['index']
    

    for index in range(0, len(train_data1)-1):
      train_data2 = train_data1.loc[0:index]
      train_data2 = train_data2[(train_data2['Buy/Hold'].str[:3] == "Buy")]


      conditions = [
                (train_data1['Buy/Hold'].loc[[index]].str[:3] == "Buy"),
                (train_data1['Buy/Hold'].loc[[index]].str[:4] == "Hold")
                 ]
      try:
       values = [train_data1['Date'].loc[[index]].map(str) + train_data1['Ticker'].loc[[index]], train_data2['Date'].iloc[-1] + train_data2['Ticker'].iloc[-1]]
      except:
       values = [train_data1['Date'].loc[[index]].map(str) + train_data1['Ticker'].loc[[index]], "str8 khusra"]
      train_data1['Transaction ID'].loc[[index]] = np.select(conditions, values, default = "")
    

     
    ###############################################
    ####           Length of Trades            ####
    ###############################################  
    
    conditions = [
         (train_data1['Buy/Hold'] == "Buy at Open")
         ]
    values = [1]
    train_data1['Length of Trade'] = np.select(conditions, values, default= "")
    train_data1['Length of Trade'] = pd.to_numeric(train_data1['Length of Trade'], errors='coerce')
    
    
    def TradeLength(i):
        conditions = [
         (train_data1['Buy/Hold'] == "Hold")
         ]
        values = [1+train_data1['Length of Trade'].shift(1)]
        train_data1['Length of Trade'] = np.select(conditions, values, default= train_data1['Length of Trade'])
        train_data1['Length of Trade'] = pd.to_numeric(train_data1['Length of Trade'], errors='coerce')
        return i
    
    TradeLength(1)
    
    for i in range(10):
        # print(i)
         TradeLength(i)
    
    # Leave highest figure to indicate length of trade
    
    conditions = [
         (train_data1['Sell/Hold'] == "Hold")
         ]
    values = [""]
    train_data1['Length of Trade'] = np.select(conditions, values, default= train_data1['Length of Trade'])
    train_data1['Length of Trade'] = pd.to_numeric(train_data1['Length of Trade'], errors='coerce')
    
    
    
    ###############################################
    ####     TakeProfit + StopLoss Columns     ####
    ###############################################  
    train_data1['n'] = n   
    train_data1['TP'] = TakeProfit     
    train_data1['SL'] = StopLoss
    train_data1['n'] = pd.to_numeric(train_data1['n'], errors='coerce')
    train_data1['TP'] = pd.to_numeric(train_data1['TP'], errors='coerce')
    train_data1['SL'] = pd.to_numeric(train_data1['SL'], errors='coerce')
    
    
    
    ###############################################
    ####        Filter for sell outcomes       ####
    ###############################################  
   
    
    conditions = [
         (train_data1['P/L%']>0),
         (train_data1['P/L%']<0)
         ]
    values = ["Profit", "Loss"]
    train_data1['Trade Outcome'] = np.select(conditions, values, default = "")

    train_data4 = train_data4.append(train_data1)  
       

###############################################
####           Iteration Outcome           ####
############################################### 

IterationOutcome = train_data4.groupby(['Ticker', 'Iteration', 'n', 'TP', 'SL', 'Trade Outcome']).size().reset_index(name='Count of Trades')
AverageLength = train_data4.groupby(['Ticker', 'Iteration', 'Trade Outcome'])['Length of Trade'].mean().reset_index(name='Average Trade Length')
AveragePL = train_data4.groupby(['Ticker', 'Iteration', 'Trade Outcome'])['P/L%'].mean().reset_index(name='Average P/L')
#print(IterationOutcome)
IterationOutcome = IterationOutcome.merge(AverageLength, how='left', on=['Ticker', 'Iteration', 'Trade Outcome'])
IterationOutcome = IterationOutcome.merge(AveragePL, how='left', on=['Ticker', 'Iteration', 'Trade Outcome'])


conditions = [
         (IterationOutcome['Trade Outcome']=="Profit") & (IterationOutcome['Trade Outcome'].shift(1)=="Loss"),
         (IterationOutcome['Trade Outcome']=="Profit") & (IterationOutcome['Trade Outcome'].shift(1)!="Loss")
         ]
 
values = [IterationOutcome['Average P/L'], IterationOutcome['Average P/L']]
IterationOutcome['Avg Profit'] = np.select(conditions, values, default = "")
IterationOutcome['Avg Profit'] = pd.to_numeric(IterationOutcome['Avg Profit'], errors='coerce')

values = [IterationOutcome['Average P/L'].shift(1), ""]
IterationOutcome['Avg Loss'] = np.select(conditions, values, default = "")
IterationOutcome['Avg Loss'] = pd.to_numeric(IterationOutcome['Avg Loss'], errors='coerce')

values = [IterationOutcome['Count of Trades']/(IterationOutcome['Count of Trades']+IterationOutcome['Count of Trades'].shift(1)), 1]
IterationOutcome['% Take Profit'] = np.select(conditions, values, default = "")

values = [(IterationOutcome['TP']/(IterationOutcome['TP']+abs(IterationOutcome['SL']))),1]
IterationOutcome['Forecasted Break Even % Take Profit'] = np.select(conditions, values, default = "")


values = [(IterationOutcome['Avg Profit']/(IterationOutcome['Avg Profit']+abs(IterationOutcome['Avg Loss']))),1]
IterationOutcome['Actual Break Even % Take Profit'] = np.select(conditions, values, default = "")


values = [((IterationOutcome['Count of Trades']*IterationOutcome['Average Trade Length'])
           +(IterationOutcome['Count of Trades'].shift(1)*IterationOutcome['Average Trade Length'].shift(1)))
          /(IterationOutcome['Count of Trades']+IterationOutcome['Count of Trades'].shift(1)), IterationOutcome['Average Trade Length']]
IterationOutcome['Average Trade Length'] = np.select(conditions, values, default = "")

values = [IterationOutcome['Count of Trades'].shift(1),0]
IterationOutcome['# Losing Trades'] = np.select(conditions, values, default = "")
#
IterationOutcome = IterationOutcome.rename(columns={'Count of Trades': '# Winning Trades'})

IterationOutcome['# Winning Trades'] = pd.to_numeric(IterationOutcome['# Winning Trades'], errors='coerce')
IterationOutcome['# Losing Trades'] = pd.to_numeric(IterationOutcome['# Losing Trades'], errors='coerce')


conditions = [
         (IterationOutcome['% Take Profit']>IterationOutcome['Actual Break Even % Take Profit'])
         ]
values = ["Yes"]
IterationOutcome['Profitable'] = np.select(conditions, values, default = "No")


IterationOutcome = IterationOutcome[(IterationOutcome["Trade Outcome"]=="Profit")]



###############################################
####        P/L Outcome ACTIONS ONLY       ####
############################################### 

conditions = [
         (train_data4['Buy/Hold']=="Buy at Open") & (train_data4['Sell/Hold'].str[:4]=="Sell"),
         (train_data4['Buy/Hold']=="Buy at Open"),
         (train_data4['Sell/Hold'].str[:4]=="Sell") 
        
         ]
values = ["Buy + Sell", "Buy", "Sell"]
train_data4['Action'] = np.select(conditions, values, default = "")

train_data4 = train_data4[(train_data4["Action"]!="")]

train_data4 = train_data4[['Date', 'Ticker', 'Transaction ID', 'Action', 'P/L%']]
train_data4 = train_data4.sort_values(by=['Date'])

PLOutput = train_data4
PLOutput = PLOutput.reset_index(drop=True)


###############################################
####  Reordering/Ranking Intraday Actions  ####
############################################### 


print(PLOutput)
for index in [15]:
#for index in range(2, len(PLOutput)-1):
 PLOutput2 = PLOutput.loc[1:index]   
 PLOutput2 = PLOutput2[PLOutput2['Action'].str[-4:]=="Sell"]
 print(PLOutput2)
 PLOutput2 = PLOutput2.groupby(['Ticker'])['P/L%'].mean()
 PLOutput2 = PLOutput2.rename("Average Ticker Return")
 PLOutputx = PLOutput.merge(PLOutput2, how='left', on='Ticker')

print(PLOutputx)



###############################################
####             P/L Outcome 2             ####
############################################### 

# Entry balance
StartingInvestment = 1000

PLOutput['Enter']=""
PLOutput['Invested']=""
PLOutput['Sold for']=""
PLOutput['P/L']=""
PLOutput['Remaining']=""
PLOutput['Portfolio']=""
PLOutput['Action Executed']=""

#for index in [148]:
for index in range(0, len(PLOutput)-1):
 PLOutput2 = PLOutput.loc[0:index]   
 print(str(index) + " : "+ str(len(PLOutput)-1))

 conditions = [
          (index == 0), 
          (index > 0)        
              ]
 try:   
  values = [StartingInvestment, PLOutput2['Remaining'].iloc[-2]]
 except:
  values = [StartingInvestment, PLOutput2['Remaining'].iloc[-1]]
 PLOutput['Enter'][[index]] = np.select(conditions, values, default = 0)
 PLOutput['Enter'][[index]] = pd.to_numeric(PLOutput['Enter'], errors='coerce')



 PLOutput4 = PLOutput2[(PLOutput2['Action'].str[-4:]=="Sell")].groupby(['Date']).count()
 PLOutput4 = PLOutput4.reset_index()
 PLOutput5 = PLOutput2.groupby(['Date']).count()
 PLOutput5 = PLOutput5.reset_index()
 PLOutput6 = PLOutput2.loc[0:index-1].groupby(['Date'])['Invested'].sum()
 PLOutput6 = PLOutput6.reset_index()
 PLOutput7 = PLOutput2.loc[0:index-1].groupby(['Date'])['Sold for'].sum()
 PLOutput7 = PLOutput7.reset_index()
 

# Invested in trade

 try:
  conditions = [
          (PLOutput4['Remaining'].iloc[-1] > 0) & (PLOutput4['Date'].iloc[-1] == PLOutput['Date'][[index]]) & (PLOutput5['Remaining'].iloc[-1] > 1) & (PLOutput5['Date'].iloc[-1] == PLOutput['Date'][[index]]) & (PLOutput6['Invested'].iloc[-1] > 0) & (PLOutput6['Date'].iloc[-1] == PLOutput['Date'][[index]]),
          (PLOutput4['Remaining'].iloc[-1] > 0) & (PLOutput4['Date'].iloc[-1] == PLOutput['Date'][[index]]) & (PLOutput5['Remaining'].iloc[-1] > 1) & (PLOutput5['Date'].iloc[-1] == PLOutput['Date'][[index]]) & (PLOutput7['Sold for'].iloc[-1] > 0) & (PLOutput7['Date'].iloc[-1] == PLOutput['Date'][[index]]) ,
          (PLOutput['Enter'][[index]] > 1) & (PLOutput['Action'][[index]].str[:3] == "Buy")     
              ]
  values = [0,0, PLOutput['Enter'][[index]] * float(1.0)]
 
 except:
  conditions = [
          (PLOutput['Enter'][[index]] > 1) & (PLOutput['Action'][[index]].str[:3] == "Buy")     
              ]    
  
  values = [PLOutput['Enter'][[index]] * float(1.0)]   
 PLOutput['Invested'][[index]] = np.select(conditions, values, default = 0)
 PLOutput['Invested'][[index]] = pd.to_numeric(PLOutput['Invested'], errors='coerce')
 
 
 
# Sold for

 try:
  PLOutput3 = PLOutput2[(PLOutput2['Action'].str[:3] == "Buy") & (PLOutput2['Transaction ID'] == PLOutput['Transaction ID'][[index]].item())]
 except:
  PLOutput3 = PLOutput2[(PLOutput2['Action'].str[:3] == "Buy")]

 conditions = [
             (PLOutput['Action'][[index]].str[-4:]=="Sell")        
             ]
 try:   
  values = [(int(1) + PLOutput['P/L%'][[index]]) * PLOutput3['Invested'].iloc[-1]]
 except:
  values = [(int(1) + PLOutput['P/L%'][[index]]) * int(0)]  
  
 PLOutput['Sold for'][[index]] = np.select(conditions, values, default = 0)
 PLOutput['Sold for'][[index]] = pd.to_numeric(PLOutput['Sold for'], errors='coerce')
 
 
# Remaining

 PLOutput['Remaining'][[index]] = PLOutput['Enter'][[index]] - PLOutput['Invested'][[index]] + PLOutput['Sold for'][[index]]

 PLOutput['Portfolio'][[index]] = PLOutput['Remaining'][[index]] + PLOutput['Invested'][[index]]

 conditions = [
          (PLOutput['Invested'][[index]] > 0),
          (PLOutput['Sold for'][[index]] > 0)     
              ]    
 
 values = [1,1] 
 PLOutput['Action Executed'][[index]]= np.select(conditions, values, default = 0)


print("################################################################")  
  
print(PLOutput)
PLOutput.to_csv(r'C:\Users\akhtam3\OneDrive - Highways England\DesktopProfitLossTracking.csv', index_label='Index')
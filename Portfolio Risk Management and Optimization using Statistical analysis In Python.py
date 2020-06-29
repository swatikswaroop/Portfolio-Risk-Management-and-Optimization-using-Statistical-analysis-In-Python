#!/usr/bin/env python
# coding: utf-8

# In[217]:


#Portfolio Risk Management and Optimization using Statistical analysis In Python
#Jupyter Notebook, Python 3, Statistics, Python Packages – Numpy, Pandas, Matplotlib
#•	Used Modern Portfolio Theory
#•	Stocks considered – Apple, Microsoft, Netflix, Amazon, Google
#•	Considered stock data from 1-1-2010 till date
#•	Individual Stock Analysis - Daily Returns, Covariance & Correlation
#•	Portfolio Analysis – Mean, Variance & Volatility
#•	Performed Monte Carlo Simulation (10000 random weight combinations)
#•	Plotted Efficient Frontier
#•	Optimized based on Sharpe Ratio


# In[218]:


import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt


# In[219]:


tickers = [ 'AAPL', 'MSFT', 'NFLX', 'AMZN', 'GOOG']
data = pd.DataFrame()
for t in tickers:
    data[t] = wb.DataReader(t, data_source='yahoo', start='01/01/2010')['Adj Close']


# In[220]:


data.head()


# <p><b> Graphical Representation of Daily Stock Prices :</b></p>

# In[221]:


(data/data.iloc[0] * 100).plot(figsize = (12,6))


# <p><b> Individual Security Returns & Risk : </b></p>

# In[222]:


daily_returns = np.log(data/data.shift(1))
daily_returns


# In[223]:


daily_returns.hist(bins=100)


# In[224]:


daily_returns.plot(figsize = (8,5))


# In[225]:


annual_returns = daily_returns.mean() * 252
annual_returns


# In[226]:


covar = daily_returns.cov() * 252
covar


# In[227]:


corr = daily_returns.corr()
corr


# <p><b>Portfolio Analysis (assigning random weights to the stocks) :</b></p>

# In[228]:


weights = np.random.random(len(tickers))
weights /= np.sum(weights)
weights


# <p><b> Expected Portfolio Return : </b></p>

# In[229]:


np.dot(annual_returns, weights)


# <p><b> Expected Portfolio Variance : </b></p>

# In[230]:


np.dot(weights.T, np.dot(covar, weights))


# <p><b> Expected Portfolio Volatility : </b></p>

# In[231]:


np.sqrt(np.dot(weights.T, np.dot(covar, weights)))


# <p><b>Monte-Carlo Simulation (generating 10,000 weight combinations) :</b></p>

# In[232]:


pfolio_returns = []
pfolio_volatilities = []
sharpe_ratio = []
stock_weights = []


# In[233]:


for x in range(10000):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    pfolio_returns.append(np.dot(daily_returns.mean()*252, weights))
    pfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov()*252, weights))))
    sharpe_ratio.append(np.dot(annual_returns, weights)/np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov()*252, weights))))
    stock_weights.append(weights)
    
pfolio_returns = np.array(pfolio_returns)
pfolio_volatilities = np.array(pfolio_volatilities)
sharpe_ratio = np.array(sharpe_ratio)
stock_weights = np.array(stock_weights)

pfolio_returns, pfolio_volatilities, sharpe_ratio, stock_weights


# In[234]:


portfolios = pd.DataFrame({'Return' : pfolio_returns,'Volatility' : pfolio_volatilities,'Sharpe Ratio' : sharpe_ratio})


# In[235]:


for counter,symbol in enumerate(tickers):
    portfolios[symbol] = [weights[counter] for weights in stock_weights]


# In[236]:


portfolios.head()


# In[237]:


max_sharpe = portfolios.iloc[portfolios['Sharpe Ratio'].idxmax()]
max_sharpe


# In[238]:


portfolios.plot.scatter(x = 'Volatility', y = 'Return', c = 'Sharpe Ratio', cmap= 'viridis', figsize = (14,8));
plt.scatter(max_sharpe[1], max_sharpe[0], marker=(5,1,0), color='r', s = 200)
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')


# In[239]:


max_sharpe.to_frame().T


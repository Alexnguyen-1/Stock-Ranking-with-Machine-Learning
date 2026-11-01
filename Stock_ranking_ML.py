import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from sklearn import ensemble, linear_model, preprocessing, pipeline, covariance
import yfinance as yf

# ---- Settings -----------------
market='SPY'
riskfree_asset='^IRX'  
start_date='2012-01-01'
end_date='2026-10-31'
horizon=21       
first_test_year=2019
train_every=5
stocks_number_each_side=10
asset_max_weight=0.15
cost_bps=10          
covariance_lookback=126
results_folder='results'
os.makedirs(results_folder,exist_ok=True)
os.makedirs('data',exist_ok=True)

sector_stocks={
    'Technology':['AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC', 'CSCO', 'ORCL', 'ADBE', 'CRM', 'QCOM', 'TXN', 'AVGO', 'IBM', 'ACN', 'INTU', 'AMAT', 'MU'],
    'Communication':['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS'],
    'Consumer Discretionary':['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'BKNG', 'TGT'],
    'Consumer Staples':['PG', 'KO', 'PEP', 'WMT', 'COST', 'MDLZ', 'CL', 'MO', 'PM'],
    'Health Care':['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'BMY', 'AMGN', 'GILD', 'MDT', 'CVS', 'DHR'],
    'Financials':['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'MA', 'V', 'SCHW', 'USB', 'BK', 'MET', 'AIG', 'COF'],
    'Industrials':['CAT', 'BA', 'HON', 'UNP', 'UPS', 'GE', 'MMM', 'LMT', 'RTX', 'DE', 'EMR', 'FDX', 'GD'],
    'Energy':['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY'],
    'Utilities':['NEE', 'DUK', 'SO', 'D', 'EXC'],
    'Materials':['LIN', 'APD', 'NEM', 'SHW'],
    'Real Estate':['AMT', 'SPG', 'PLD'],
}
stocks=[ticker for tickers in sector_stocks.values() for ticker in tickers]
sector_map={ticker:sector for sector,tickers in sector_stocks.items() for ticker in tickers}

# ---- Data -----------------
if os.path.exists('data/prices.pkl'):
    data=pd.read_pickle('data/prices.pkl')
else:
    data=yf.download(stocks+[market,riskfree_asset],start=start_date,end=end_date,auto_adjust=True)
    data.to_pickle('data/prices.pkl')
riskfree=data['Close'][riskfree_asset].ffill()
data=data.drop(columns=riskfree_asset,level=1)

# Reformat to one row per stock per day
df=data.stack(level=1).reset_index().rename(columns={'level_1':'Ticker'})
df=df[['Date','Ticker','High','Low','Close','Volume']]

# Market daily return, forward return and the T-bill return over the same 21 days
spy=df[df['Ticker']==market][['Date','Close']].rename(columns={'Close':'SPY_Close'}).sort_values('Date')
spy['SPY_Return']=spy['SPY_Close']/spy['SPY_Close'].shift(1)-1
spy['SPY_Forward_Return']=spy['SPY_Close'].shift(-horizon)/spy['SPY_Close']-1
spy['Riskfree_Period']=(1+riskfree.reindex(spy['Date']).ffill().values/100)**(horizon/252)-1
trading_days=spy['Date'].values
df=pd.merge(df[df['Ticker']!=market],spy.drop(columns='SPY_Close'),on='Date',how='left')
df=df.sort_values(['Ticker','Date']).reset_index(drop=True)

# ---- Independent variables -----------------
def rolling_func(column,window,stat='mean'):
    # Rolling statistic calculated separately for each stock
    return(df.groupby('Ticker')[column].rolling(window).agg(stat).reset_index(level=0,drop=True))

df['Daily_Return']=df['Close']/df.groupby('Ticker')['Close'].shift(1)-1
# 5, 21 and 63 trading-day momentum, plus 12-1 month momentum
df['Momentum_5']=df['Close']/df.groupby('Ticker')['Close'].shift(5)-1
df['Momentum_21']=df['Close']/df.groupby('Ticker')['Close'].shift(21)-1
df['Momentum_63']=df['Close']/df.groupby('Ticker')['Close'].shift(63)-1
df['Momentum_12_1']=df.groupby('Ticker')['Close'].shift(21)/df.groupby('Ticker')['Close'].shift(252)-1
# Price vs. 21 and 200-day moving average
df['MA_Ratio']=df['Close']/rolling_func('Close',21)
df['MA_Ratio_200']=df['Close']/rolling_func('Close',200)
# Volatility, biggest daily gain in the past month and average high-low range
df['Volatility_21']=rolling_func('Daily_Return',21,'std')
df['Volatility_63']=rolling_func('Daily_Return',63,'std')
df['Max_Return_21']=rolling_func('Daily_Return',21,'max')
df['Daily_Range']=(df['High']-df['Low'])/df['Close']
df['High-low_range']=rolling_func('Daily_Range',21)
# Volume ratio, volume momentum (21-day vs 63-day average) and dollar volume
df['Volume']=df['Volume'].replace(0,np.nan)
df['Volume_Ratio']=df['Volume']/rolling_func('Volume',21)
df['Volume_Momentum']=rolling_func('Volume',21)/rolling_func('Volume',63)-1
df['Daily_Dollar_Volume']=df['Close']*df['Volume']
df['Dollar_Volume']=np.log(rolling_func('Daily_Dollar_Volume',63))
# 63-day beta to SPY
df['Return_x_SPY']=df['Daily_Return']*df['SPY_Return']
df['SPY_Return_Squared']=df['SPY_Return']**2
covariance_63=rolling_func('Return_x_SPY',63)-rolling_func('Daily_Return',63)*rolling_func('SPY_Return',63)
variance_63=rolling_func('SPY_Return_Squared',63)-rolling_func('SPY_Return',63)**2
df['Beta_63']=covariance_63/variance_63
# Beta used for hedging and for the residual return: 252-day beta shrunk toward 1 (Blume adjustment).
# Hedging with the raw 63-day beta left the beta-neutral portfolios with leftover market exposure
covariance_252=rolling_func('Return_x_SPY',252)-rolling_func('Daily_Return',252)*rolling_func('SPY_Return',252)
variance_252=rolling_func('SPY_Return_Squared',252)-rolling_func('SPY_Return',252)**2
df['Hedge_Beta']=0.67*covariance_252/variance_252+0.33

# Daily returns table
daily_returns=df.pivot(index='Date',columns='Ticker',values='Daily_Return')

# ---- Target variable -----------------
df['Forward_Return']=df.groupby('Ticker')['Close'].shift(-horizon)/df['Close']-1

independent_vars=['Momentum_5',
'Momentum_21',
'Momentum_63',
'Momentum_12_1',
'MA_Ratio',
'MA_Ratio_200',
'Volatility_21',
'Volatility_63',
'Max_Return_21',
'High-low_range',
'Volume_Ratio',
'Volume_Momentum',
'Dollar_Volume',
'Beta_63',
]
# Clean data and keep dates with at least 30 stocks
df[independent_vars]=df[independent_vars].replace([np.inf,-np.inf],np.nan)
df=df.dropna(subset=independent_vars+['Hedge_Beta']).copy()
df=df[df.groupby('Date')['Ticker'].transform('size')>=30].copy()
df['Sector']=df['Ticker'].map(sector_map)

# Residual return: exclude the market part from total return, then the sector average
df['Residual_Return']=df['Forward_Return']-df['Hedge_Beta']*df['SPY_Forward_Return']
df['Residual_Return']=df['Residual_Return']-df.groupby(['Date','Sector'])['Residual_Return'].transform('mean')

# Rank features and target against the other stocks on the same day
for var in independent_vars:
    df[var]=df.groupby('Date')[var].rank(pct=True)-0.5
df['Target']=df.groupby('Date')['Residual_Return'].rank(pct=True)-0.5
df['Day']=np.searchsorted(trading_days,df['Date'].values)

# Classic factor signals as baselines
df['Signal_Momentum']=df['Momentum_12_1']
df['Signal_Reversal']=-df['Momentum_21']
df['Signal_Low_Vol']=-df['Volatility_63']
print(f'Data: {len(df):,} rows, {df["Ticker"].nunique()} stocks, {df["Date"].min().date()} to {df["Date"].max().date()}')

# ---- Models -----------------
def ic_func(dates,prediction,actual):
    # Information Coefficient: rank correlation between prediction and actual return, one value per date
    temp=pd.DataFrame({'Date':np.asarray(dates),'Prediction':np.asarray(prediction,float),'Actual':np.asarray(actual,float)}).dropna()
    temp[['Prediction','Actual']]=temp.groupby('Date')[['Prediction','Actual']].rank()
    temp[['Prediction','Actual']]=temp[['Prediction','Actual']]-temp.groupby('Date')[['Prediction','Actual']].transform('mean')
    temp['PA']=temp['Prediction']*temp['Actual']
    temp['PP']=temp['Prediction']**2
    temp['AA']=temp['Actual']**2
    sums=temp.groupby('Date')[['PA','PP','AA']].sum()
    return(sums['PA']/np.sqrt(sums['PP']*sums['AA']))

def ridge_func(degree,alpha):
    Input=[('scaler',preprocessing.StandardScaler()),('poly_features',preprocessing.PolynomialFeatures(degree=degree,include_bias=False)),('Ridge',linear_model.Ridge(alpha=alpha))]
    return(pipeline.Pipeline(Input))

def tune_ridge_func(train_df):
    # Grid search with time-ordered folds. Each fold trains only on days before the validation block,
    # with a 21-day gap so no training target falls into the validation period
    days=train_df['Day'].values
    dates=train_df['Date'].values
    x=train_df[independent_vars].values
    y=train_df['Target'].values
    folds=np.array_split(np.unique(days),5)
    best_params,best_ic=None,-np.inf
    for degree in [1,2]:
        for alpha in [1,10,100,1000,10000,100000]:
            fold_ic=[]
            for fold in folds[1:]:
                train_mask=days<=fold[0]-horizon-1
                valid_mask=(days>=fold[0])&(days<=fold[-1])
                model=ridge_func(degree,alpha).fit(x[train_mask],y[train_mask])
                fold_ic.append(ic_func(dates[valid_mask],model.predict(x[valid_mask]),y[valid_mask]).mean())
            if np.mean(fold_ic)>best_ic:
                best_params,best_ic=(degree,alpha),np.mean(fold_ic)
    return(best_params,best_ic)

# ---- Walk-forward training -----------------
# Each test year gets new models trained only on data from before that year.
# Embargo: a training row's target looks 21 days ahead, so training rows must end 22+ days before the test year
test_years=sorted(year for year in df['Date'].dt.year.unique() if year>=first_test_year)
train_pool=df[(df['Day']%train_every==0)&df['Target'].notna()]
test_df=df[df['Date'].dt.year>=first_test_year].copy()
training_log=[]
importance_list=[]
print('\nWalk-forward training:')
for year in test_years:
    test_mask=test_df['Date'].dt.year==year
    first_day=test_df.loc[test_mask,'Day'].min()
    train_df=train_pool[train_pool['Day']<=first_day-horizon-1]
    x_train=train_df[independent_vars].values
    y_train=train_df['Target'].values
    x_test=test_df.loc[test_mask,independent_vars].values
    (degree,alpha),cv_ic=tune_ridge_func(train_df)
    Best_ridge=ridge_func(degree,alpha).fit(x_train,y_train)
    rf=ensemble.RandomForestRegressor(n_estimators=200,max_depth=6,min_samples_leaf=200,max_features=0.5,n_jobs=-1,random_state=42)
    rf.fit(x_train,y_train)
    test_df.loc[test_mask,'Prediction_Ridge']=Best_ridge.predict(x_test)
    test_df.loc[test_mask,'Prediction_RF']=rf.predict(x_test)
    importance_list.append(pd.Series(rf.feature_importances_,index=independent_vars,name=year))
    training_log.append({'Test Year':year,'Train From':train_df['Date'].min().date(),'Train To':train_df['Date'].max().date(),
                         'Train Rows':len(train_df),'Ridge Degree':degree,'Ridge Alpha':alpha,'CV IC':round(cv_ic,4)})
    print(f'  {year}: trained on {train_df["Date"].min().date()} to {train_df["Date"].max().date()}, Ridge degree={degree} alpha={alpha}, CV IC={cv_ic:.4f}')

# Monthly rebalance dates with every 21st trading day of the test period
valid_days=np.sort(test_df.loc[test_df['Forward_Return'].notna(),'Day'].unique())
rebalance_days=np.intersect1d(np.arange(valid_days[0],valid_days[-1]+1,horizon),valid_days)
monthly_df=test_df[test_df['Day'].isin(rebalance_days)&test_df['Forward_Return'].notna()].copy()
print(f'\nTest period: {monthly_df["Date"].nunique()} monthly rebalances, {monthly_df["Date"].min().date()} to {monthly_df["Date"].max().date()}')

# ---- Signal quality (IC) -----------------
signals={'Random Forest':'Prediction_RF',
         'Ridge':'Prediction_Ridge',
         'Momentum 12-1':'Signal_Momentum',
         'Reversal 1M':'Signal_Reversal',
         'Low Volatility':'Signal_Low_Vol'}
ic_df=pd.DataFrame({name:ic_func(monthly_df['Date'],monthly_df[col],monthly_df['Residual_Return']) for name,col in signals.items()})

def newey_west_tstat_func(series,lags=6):
    # t-stat adjusted for autocorrelation since monthly ICs of slow-changing signals aren't independent
    x=series.dropna().values
    x_demeaned=x-x.mean()
    variance=np.sum(x_demeaned**2)/len(x)
    for lag in range(1,lags+1):
        variance+=2*(1-lag/(lags+1))*np.sum(x_demeaned[lag:]*x_demeaned[:-lag])/len(x)
    return(x.mean()/np.sqrt(variance/len(x)))

ic_summary=pd.DataFrame({'Mean IC':ic_df.mean(),'IC Std':ic_df.std(),'t-stat':ic_df.mean()/ic_df.std()*np.sqrt(ic_df.count()),
                         't-stat (NW)':ic_df.apply(newey_west_tstat_func),'% Months IC>0':(ic_df>0).mean()})
ic_summary.index.name='Signal'
ic_by_year=ic_df.groupby(ic_df.index.year).mean()
ic_by_year.index.name='Year'

# Single-feature IC, before and after removing beta and sector (shows why the residual target was needed)
feature_ic=pd.DataFrame({'IC vs Raw Return':{var:ic_func(monthly_df['Date'],monthly_df[var],monthly_df['Forward_Return']).mean() for var in independent_vars},
                         'IC vs Residual Return':{var:ic_func(monthly_df['Date'],monthly_df[var],monthly_df['Residual_Return']).mean() for var in independent_vars}})
feature_ic.index.name='Feature'

# Quintile analysis: average residual return per prediction quintile
quintile_list={}
for name in ['Random Forest','Ridge']:
    col=signals[name]
    monthly_df['Quintile']=np.ceil(monthly_df.groupby('Date')[col].rank(pct=True)*5).clip(1,5).astype(int)
    quintile_list[name]=monthly_df.groupby(['Date','Quintile'])['Residual_Return'].mean().groupby('Quintile').mean()
quintiles=pd.DataFrame(quintile_list)

# ---- Portfolio backtest -----------------
def beta_neutral_weights_func(long_betas,short_betas):
    # Equal weight inside each side, then size the sides so long beta = short beta (gross exposure stays 100%)
    long_beta=np.mean(long_betas)
    short_beta=np.mean(short_betas)
    long_size=short_beta/(long_beta+short_beta)
    short_size=long_beta/(long_beta+short_beta)
    return(np.concatenate([np.full(len(long_betas),long_size/len(long_betas)),np.full(len(short_betas),-short_size/len(short_betas))]))

# Sharpe ratio calculation function
def sharpe_ratio_func(weights,expected_returns,covariance_matrix):
    portfolio_return=np.dot(weights,expected_returns)
    portfolio_risk=np.dot(np.dot(weights,covariance_matrix),weights)
    sharpe_ratio=portfolio_return/np.sqrt(portfolio_risk)
    return(-sharpe_ratio)

def max_sharpe_weights_func(expected_returns,covariance_matrix,betas,n_long):
    # Maximize Sharpe ratio with portfolio beta = 0 and 100% gross exposure
    initial_weights=beta_neutral_weights_func(betas[:n_long],betas[n_long:])
    constraints=(
        {'type':'eq','fun':lambda w: np.sum(w[:n_long])-np.sum(w[n_long:])-1},  # gross exposure = 100%
        {'type':'eq','fun':lambda w: np.dot(w,betas)}                            # portfolio beta = 0
    )
    limits=[(0,asset_max_weight)]*n_long+[(-asset_max_weight,0)]*(len(betas)-n_long)
    optimization=optimize.minimize(sharpe_ratio_func,initial_weights,args=(expected_returns,covariance_matrix),method='SLSQP',bounds=limits,constraints=constraints)
    if optimization.success:
        return(optimization.x)
    return(initial_weights)

def backtest_func(signal,weighting):
    previous_weights=pd.Series(dtype=float)
    results=[]
    for date,month_df in monthly_df.groupby('Date'):
        month_df=month_df.dropna(subset=[signal]).sort_values(signal,ascending=False)
        top_long_stocks=month_df.head(stocks_number_each_side)
        top_short_stocks=month_df.tail(stocks_number_each_side)
        portfolio_tickers=top_long_stocks['Ticker'].tolist()+top_short_stocks['Ticker'].tolist()
        betas=np.clip(np.concatenate([top_long_stocks['Hedge_Beta'].values,top_short_stocks['Hedge_Beta'].values]),0.2,None)  # floor at 0.2 so leg sizes stay sensible
        if weighting=='beta_neutral':
            weights=beta_neutral_weights_func(betas[:stocks_number_each_side],betas[stocks_number_each_side:])
        else:
            historical_returns=daily_returns.loc[:date,portfolio_tickers].tail(covariance_lookback).fillna(0)
            covariance_matrix=covariance.LedoitWolf().fit(historical_returns.values).covariance_
            Expected_returns_total=np.concatenate([top_long_stocks[signal].values,top_short_stocks[signal].values])
            weights=max_sharpe_weights_func(Expected_returns_total,covariance_matrix,betas,stocks_number_each_side)
        weights=pd.Series(weights,index=portfolio_tickers)
        # Excess stock return over T-bill return for the same month
        excess_returns=pd.concat([top_long_stocks,top_short_stocks]).set_index('Ticker')['Forward_Return']-month_df['Riskfree_Period'].iloc[0]
        gross_return=(weights*excess_returns).sum()
        turnover=weights.sub(previous_weights,fill_value=0).abs().sum()
        cost=turnover*cost_bps/10000
        results.append({'Date':date,'Gross':gross_return,'Net':gross_return-cost,'Turnover':turnover})
        previous_weights=weights
    return(pd.DataFrame(results).set_index('Date'))

def performance_func(returns,spy_excess,turnover=None):
    # Annualized stats for monthly (21-day) excess returns, plus alpha and beta from a regression on SPY
    value=(1+returns).cumprod()
    regression=stats.linregress(spy_excess,returns)
    return({'Ann. Excess Return':value.iloc[-1]**(12/len(returns))-1,
            'Ann. Volatility':returns.std()*np.sqrt(12),
            'Sharpe Ratio':returns.mean()/returns.std()*np.sqrt(12),
            'Max Drawdown':(value/value.cummax()-1).min(),
            'Beta to SPY':regression.slope,
            'Alpha (annual)':regression.intercept*12,
            'Alpha t-stat':regression.intercept/regression.intercept_stderr,
            'Avg SPY-up Month':returns[spy_excess>0].mean(),
            'Avg SPY-down Month':returns[spy_excess<=0].mean(),
            'Avg Turnover':np.nan if turnover is None else turnover.mean()})

# Main strategy Random Forest with beta-neutral compared against other strategies
strategies={'Random Forest':('Prediction_RF','beta_neutral'),
            'Random Forest (max Sharpe)':('Prediction_RF','max_sharpe'),
            'Ridge':('Prediction_Ridge','beta_neutral'),
            'Momentum 12-1':('Signal_Momentum','beta_neutral'),
            'Reversal 1M':('Signal_Reversal','beta_neutral'),
            'Low Volatility':('Signal_Low_Vol','beta_neutral')}
backtests={}
for name,(signal,weighting) in strategies.items():
    backtests[name]=backtest_func(signal,weighting)

spy_monthly=monthly_df.groupby('Date')[['SPY_Forward_Return','Riskfree_Period']].first()
spy_excess=spy_monthly['SPY_Forward_Return']-spy_monthly['Riskfree_Period']
performance={name:performance_func(bt['Net'],spy_excess.loc[bt.index],bt['Turnover']) for name,bt in backtests.items()}
performance['SPY (buy & hold)']=performance_func(spy_excess,spy_excess)
performance_df=pd.DataFrame(performance).T
performance_df.index.name='Strategy'
monthly_returns=pd.DataFrame({name:bt['Net'] for name,bt in backtests.items()})
monthly_returns['SPY (buy & hold)']=spy_excess
monthly_returns.index.name='Date'

# ---- Save results -----------------
performance_df.to_csv(f'{results_folder}/backtest_summary.csv')
monthly_returns.to_csv(f'{results_folder}/monthly_excess_returns.csv')
ic_summary.to_csv(f'{results_folder}/ic_summary.csv')
ic_by_year.to_csv(f'{results_folder}/ic_by_year.csv')
feature_ic.to_csv(f'{results_folder}/feature_ic.csv')
quintiles.to_csv(f'{results_folder}/quintile_returns.csv')
pd.DataFrame(training_log).to_csv(f'{results_folder}/walk_forward_log.csv',index=False)
rf_importance=pd.concat(importance_list,axis=1)
rf_importance.index.name='Feature'
rf_importance.to_csv(f'{results_folder}/rf_feature_importance.csv')

# ---- Charts -----------------
sns.set_style('whitegrid')
colors={'Random Forest':'#2a78d6',
        'Random Forest (max Sharpe)':'#eb6834',
        'Ridge':'#1baf7a',
        'Momentum 12-1':'#6250d6',
        'Reversal 1M':'#eda100',
        'Low Volatility':'#e34948',
        'SPY (buy & hold)':'#898781'}

# 1. Growth of $1 (excess return over T-bills, after costs)
plt.figure(figsize=(11,6))
for name in strategies:
    plt.plot((1+backtests[name]['Net']).cumprod(),color=colors[name],linewidth=3 if name=='Random Forest' else 1.8,label=name)
plt.plot((1+spy_excess).cumprod(),color=colors['SPY (buy & hold)'],linewidth=2,linestyle='--',label='SPY (buy & hold)')
plt.axhline(1,color='black',linewidth=0.8)
plt.title(f'Growth of $1 above T-bills, after {cost_bps} bps costs (beta-neutral long-short)',loc='left',fontsize=13)
plt.ylabel('Portfolio value ($)')
plt.legend(frameon=False,fontsize=9,ncol=2)
plt.tight_layout()
plt.savefig(f'{results_folder}/equity_curves.png',dpi=150)
plt.close()

# 2. Average monthly return when SPY goes up vs down
up_down=pd.DataFrame({'SPY up months':performance_df.loc[['Random Forest','SPY (buy & hold)'],'Avg SPY-up Month'],
                      'SPY down months':performance_df.loc[['Random Forest','SPY (buy & hold)'],'Avg SPY-down Month']})*100
x_positions=np.arange(2)
plt.figure(figsize=(7,5))
plt.bar(x_positions-0.2,up_down.loc['Random Forest'],width=0.38,color=colors['Random Forest'],label='Random Forest')
plt.bar(x_positions+0.2,up_down.loc['SPY (buy & hold)'],width=0.38,color=colors['SPY (buy & hold)'],label='SPY (buy & hold)')
plt.axhline(0,color='black',linewidth=0.8)
plt.xticks(x_positions,up_down.columns)
plt.title('Average monthly excess return: up vs down markets',loc='left',fontsize=13)
plt.ylabel('Avg 21-day excess return (%)')
plt.legend(frameon=False,fontsize=9)
plt.tight_layout()
plt.savefig(f'{results_folder}/up_down_months.png',dpi=150)
plt.close()

# 3. Cumulative IC vs residual return
plt.figure(figsize=(11,6))
for name in signals:
    plt.plot(ic_df[name].cumsum(),color=colors[name],linewidth=3 if name=='Random Forest' else 1.8,label=f'{name} (mean {ic_df[name].mean():+.3f})')
plt.axhline(0,color='black',linewidth=0.8)
plt.title('Cumulative Information Coefficient (vs residual return)',loc='left',fontsize=13)
plt.ylabel('Cumulative monthly IC')
plt.legend(frameon=False,fontsize=9)
plt.tight_layout()
plt.savefig(f'{results_folder}/cumulative_ic.png',dpi=150)
plt.close()

# 4. Quintile returns
quintiles_pct=quintiles*100
x_positions=np.arange(1,6)
plt.figure(figsize=(8,5))
plt.bar(x_positions-0.2,quintiles_pct['Random Forest'],width=0.38,color=colors['Random Forest'],label='Random Forest')
plt.bar(x_positions+0.2,quintiles_pct['Ridge'],width=0.38,color=colors['Ridge'],label='Ridge')
plt.axhline(0,color='black',linewidth=0.8)
plt.xticks(x_positions,['Q1\n(predicted worst)','Q2','Q3','Q4','Q5\n(predicted best)'])
plt.title('Average residual return by prediction quintile',loc='left',fontsize=13)
plt.ylabel('Avg 21-day residual return (%)')
plt.legend(frameon=False,fontsize=9)
plt.tight_layout()
plt.savefig(f'{results_folder}/quintile_returns.png',dpi=150)
plt.close()

# 5. Feature IC before and after removing beta and sector
feature_ic_sorted=feature_ic.sort_values('IC vs Residual Return')
y_positions=np.arange(len(feature_ic_sorted))
plt.figure(figsize=(9,7))
plt.barh(y_positions+0.2,feature_ic_sorted['IC vs Raw Return'],height=0.38,color='#898781',label='vs raw return')
plt.barh(y_positions-0.2,feature_ic_sorted['IC vs Residual Return'],height=0.38,color='#2a78d6',label='vs residual return (beta and sector removed)')
plt.axvline(0,color='black',linewidth=0.8)
plt.yticks(y_positions,feature_ic_sorted.index)
plt.title('Single-feature IC, before and after removing beta and sector',loc='left',fontsize=13)
plt.xlabel('Mean monthly IC')
plt.legend(frameon=False,fontsize=9)
plt.tight_layout()
plt.savefig(f'{results_folder}/feature_ic.png',dpi=150)
plt.close()

# ---- Print results -----------------
pd.set_option('display.width',250)
pd.set_option('display.max_columns',30)
print('\nSIGNAL QUALITY (monthly IC vs residual return)')
print(ic_summary.round(3).to_string())
print('\nMEAN IC BY YEAR')
print(ic_by_year.round(3).to_string())
print('\nSINGLE-FEATURE IC')
print(feature_ic.sort_values('IC vs Residual Return',ascending=False).round(4).to_string())
print('\nQUINTILE ANALYSIS (avg 21-day residual return, %)')
print((quintiles*100).round(2).to_string())
print(f'\nPORTFOLIO BACKTEST (excess return over T-bills, net of {cost_bps} bps costs, {stocks_number_each_side} long / {stocks_number_each_side} short)')
printed_df=performance_df.copy()
for col in ['Ann. Excess Return','Ann. Volatility','Max Drawdown','Alpha (annual)','Avg SPY-up Month','Avg SPY-down Month']:
    printed_df[col]=printed_df[col].map('{:.1%}'.format)
for col in ['Sharpe Ratio','Beta to SPY','Alpha t-stat','Avg Turnover']:
    printed_df[col]=printed_df[col].map(lambda x: '' if pd.isna(x) else f'{x:.2f}')
print(printed_df.to_string())
print(f'\nResults saved to {results_folder}/')

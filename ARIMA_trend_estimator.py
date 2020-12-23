import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pandas_datareader.data as pdr
from scipy.optimize import minimize
import datetime
import matplotlib.pyplot as plt
from matplotlib import ticker
import pyflux as pf
from scipy import stats
from arch.unitroot import ADF



# FE 630 Final project
# Group member: Minghao Kang, Yuwen Jin, Shariz Bheda

preset_cov_period, preset_rho_period = 90, 150
add_back_period = max(preset_cov_period, preset_rho_period) * 1.2

'# data prepare: for FF model'
FFdata = pd.read_csv('data/F-F_Research_Data_Factors_daily.CSV', sep=",")
FFdata.columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RF']
FFdata[['Mkt-RF', 'SMB', 'HML', 'RF']] = np.array(FFdata[['Mkt-RF', 'SMB', 'HML', 'RF']]) / 100
FF_start = datetime.datetime.strptime(str(20070301), '%Y%m%d') - datetime.timedelta(days=add_back_period - 1)
FF_use = FFdata[(FFdata['date'] > int(FF_start.strftime("%Y%m%d"))) & (FFdata['date'] <= 20200709)]

# download ETF data
ETF = "FXE,EWJ,GLD,QQQ,SPY,SHV,DBA,USO,XBI,ILF,EPP,FEZ"
yf.pdr_override()
start = datetime.datetime.strptime("2007-03-01", '%Y-%m-%d')
start = start - datetime.timedelta(days=add_back_period)
start_formatted = start.strftime("%Y-%m-%d")
ETF_price = pdr.get_data_yahoo(ETF, start=start_formatted, end="2020-07-10", interval='1d').dropna()['Adj Close']
# Find daily return of Each ETF
daily_return = ETF_price.pct_change(1).dropna()

total_df = daily_return
difference = len(FF_use) - len(total_df)
total_df['Mkt-RF'] = FF_use['Mkt-RF'].values[difference:]

total_df['Mkt-RF'] = FF_use['Mkt-RF'].values[difference:]
total_df['SMB'] = FF_use['SMB'].values[difference:]
total_df['HML'] = FF_use['HML'].values[difference:]
total_df['RF'] = FF_use['RF'].values[difference:]

# FF model
# try to find 3 factor's coef first
tickers = ['FXE', 'EWJ', 'GLD', 'QQQ', 'SPY', 'SHV', 'DBA', 'USO', 'XBI', 'ILF', 'EPP', 'FEZ']


class Strategy:
	def __init__(self, symbols, start_date, end_date):
		self.symbols = symbols
		self.date = end_date
		self.CAPM_date = start_date
		self.n = len(self.symbols)
		self.df = total_df[total_df.index > datetime.datetime.strptime(start_date, '%Y-%m-%d')]
		self.df = self.df[self.df.index < datetime.datetime.strptime(end_date, '%Y-%m-%d')]
		self.Sigma = None
		self.rho = None
		self.rho_period = None

	def FF_model(self, days, date, ticker):
		dif = (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.datetime.strptime("2007-01-11", '%Y-%m-%d')).days
		if days > dif:
			days = dif
		period = total_df[total_df.index < date][-days:]
		Y = period[ticker].values - period['RF'].values
		# print(len(Y))
		Y = np.array(Y).reshape(len(Y), 1)
		mkt_rf = np.array(period['Mkt-RF']).reshape(len(Y), 1)
		SMB = np.array(period['SMB']).reshape(len(Y), 1)
		HML = np.array(period['HML']).reshape(len(Y), 1)
		X = np.hstack((mkt_rf, SMB, HML))
		a = LinearRegression()
		regression = a.fit(X, Y.flatten())
		beta3, bs, bv = regression.coef_
		alpha = regression.intercept_
		return beta3, bs, bv, alpha

	def find_sig(self, days, date, ticker1, ticker2):
		dif = (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.datetime.strptime("2007-01-11", '%Y-%m-%d')).days
		if days > dif:
			days = dif
		period = total_df[total_df.index < self.date][-dif:]
		beta3_i, bs_i, bv_i = self.FF_model(dif, date, ticker1)[:-1]
		beta3_j, bs_j, bv_j = self.FF_model(dif, date, ticker2)[:-1]
		covbibj = beta3_i * beta3_j * np.var(period['Mkt-RF']) + bs_i * bs_j * np.var(period['SMB']) + \
				  bv_i * bv_j * np.var(period['HML']) + \
				  (beta3_i * bs_j + beta3_j * bs_i) * np.cov(period['Mkt-RF'], period['SMB'])[0][1] + \
				  (beta3_i * bv_j + beta3_j * bv_i) * np.cov(period['Mkt-RF'], period['SMB'])[0][1] + (
						  bs_i * bv_j + bs_j * bv_i) * np.cov(period['SMB'], period['HML'])[0][1]

		return np.sum(covbibj)

	def Find_cov_mat(self, days, date):
		n = 12
		cov_mat = np.empty((n, n,))
		cov_mat[:] = np.nan
		for i in range(n):
			for j in range(n):
				cov_mat[i - 1, j - 1] = self.find_sig(days, date, self.symbols[i], self.symbols[j])
		print(cov_mat)

		return cov_mat

	def CAPM(self, symbol, days):
		period = total_df[total_df.index < self.CAPM_date][-days:]
		rho = np.array(period[symbol].values)
		rm = np.array(period['Mkt-RF']) + np.array(period['RF'])
		return np.cov(rho, rm)[0, 1] / np.var(rm)

	def find_para_lookback(self, cov_period, rho_period):
		"""
		:param cov_period: the length of period to look for the covariance
		:param rho_period: the length of period to look for the expected return
		:return: covariance and expected return
		"""
		dif = (datetime.datetime.strptime(self.CAPM_date, '%Y-%m-%d') -
			   datetime.datetime.strptime("2007-01-11", '%Y-%m-%d')).days
		if cov_period > dif:
			cov_period = dif
		if rho_period > dif:
			rho_period = dif
		# period = total_df[total_df.index < self.CAPM_date][-cov_period:]
		# self.Sigma = period[self.symbols].cov()
		self.Sigma = self.Find_cov_mat(cov_period, self.CAPM_date)
		period = total_df[total_df.index < self.CAPM_date][-rho_period:]
		self.rho_period = rho_period
		self.rho = period[self.symbols].apply(np.mean, axis=0)

		return self.Sigma, self.rho

	def estimator(self, back_period, end_data, ticker, sigma, h):
		"""
		:param back_period: how much days you're looking back to find the rho
		:param end_data: from which data you start to look back
		:param ticker: symbol of a single asset
		:param sigma: the volatility of the stock
		:param h: how much steps you want to predict
		:return: expected return in 'coming' h days
		"""

		train = total_df[total_df.index < end_data][-back_period:][[ticker, 'Mkt-RF', 'SMB', 'HML', 'RF']]
		train.columns = ['y1', 'x1', 'x2', 'x3', 'y2']
		test = total_df[total_df.index >= end_data][:h][[ticker, 'Mkt-RF', 'SMB', 'HML', 'RF']]
		print("test")
		print(test)
		test.columns = ['y1', 'x1', 'x2', 'x3', 'y2']
		test[['x1', 'x2', 'x3']] = train[['x1', 'x2', 'x3']].iloc[-h:, :].values

		R_i = total_df[total_df.index < end_data][-back_period:][ticker]
		r_f = total_df[total_df.index < end_data][-back_period:]['RF']
		F = total_df[total_df.index < end_data][-back_period:][['Mkt-RF', 'SMB', 'HML']]

		lm = LinearRegression()
		lm.fit(np.array(R_i - r_f).reshape(len(R_i), 1), F)
		coefficients = lm.coef_.flatten()

		model1 = pf.ARIMAX(data=train, formula="x1~1", ar=5, ma=5, integ=0)
		model1.fit("MLE")
		prediction1 = model1.predict(h=h, oos_data=test).values.flatten()  # Mkt-RF

		model2 = pf.ARIMAX(data=train, formula="x2~1", ar=5, ma=5, integ=0)
		model2.fit("MLE")
		prediction2 = model2.predict(h=h, oos_data=test).values.flatten()  # SMB

		model3 = pf.ARIMAX(data=train, formula="x3~1", ar=5, ma=5, integ=0)
		model3.fit("MLE")
		prediction3 = model3.predict(h, test).values.flatten()  # HML

		residual = np.random.normal(0, sigma, h)
		expected_rho = test['y2'] + coefficients.dot([prediction1, prediction2, prediction3]) + residual

		return np.mean(expected_rho)

	def find_wei(self, CAPM_period, betaT, omega_p):
		lamb = 0.0001
		beta_list = []
		for symbol in self.symbols:
			beta_list.append(self.CAPM(symbol, CAPM_period))

		fun = lambda x: lamb * (x - omega_p).dot(self.Sigma).dot(x - omega_p) - self.rho.dot(x)  # constrain function
		cons = ({'type': 'eq', 'fun': lambda x: np.array(beta_list).dot(x) - betaT},
				{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Initial Wealth Fully Invested
				)
		x0 = np.array([1 / self.n] * self.n)  # Initial guess
		res = minimize(fun, x0, method='SLSQP', constraints=cons, bounds=[(-2, 2)] * 12)

		return res.x

	def test(self, CAPM_period, betaT):
		days = self.df.index
		prices_raw = self.df.copy()
		df = self.df[self.symbols].copy()
		# df = total_df[tickers].copy()
		# to make groups of data in order to find prices of every single sticks
		weekdays = []
		stamp = []
		temp_day = days[0]
		end_date = days[-1]
		while temp_day < end_date:
			if temp_day in days:
				weekdays.append(int(temp_day.strftime("%w")))
			temp_day += datetime.timedelta(days=1)

		for i in range(len(weekdays) - 1):
			if weekdays[i + 1] < weekdays[i]:
				stamp.append(i)

		n_period = len(stamp)
		omega_list = [[1 / self.n] * self.n]
		# omega_list = [[1 / 12] * 12]
		for i in range(n_period):
			print('period' + str(i))
			for m in range(self.n):
				# back_period, end_data, df, ticker, sigma, h
				self.rho[m] = self.estimator(self.rho_period, days[stamp[i]],
											 self.symbols[m], np.diag(self.Sigma)[m], 5)
			omega = self.find_wei(CAPM_period, betaT, omega_list[i])
			omega_list.append(omega)

		prices_raw['help_col'] = range(len(prices_raw))
		prices_raw['help_col_1'] = range(len(prices_raw))

		# find all mondays
		for i in stamp:
			prices_raw.loc[prices_raw['help_col'] >= i, ['help_col_1']] = i
		prices_raw.loc[prices_raw['help_col'] <= stamp[0], ['help_col_1']] = 0

		asset = []
		for i in range(n_period):
			returns = df[prices_raw['help_col_1'] == stamp[i]]
			temp = returns.dot(omega_list[i])
			for a in temp:
				asset.append(a)

		return asset


def maximum_10_drawdown(returns):
	max_dd = 0
	for i in range(len(returns) - 10):
		ten_day = returns[i:i + 10]
		max = np.max(ten_day)
		min = np.min(ten_day)
		dd_10 = (max - min) / min
		max_dd = np.max(np.array([dd_10, max_dd]))
	return max_dd


def sample_para_VaR(data, size):
	sample = np.random.choice(data, size, replace=True)
	VaR_5 = np.percentile(sample, 5)
	return VaR_5


def insert_series(series):
	new_series = [i for i in range(len(series))]
	n = len(series)
	steps = int(n / 5)
	ranges = list(np.arange(0, len(series), step=steps))
	for i in range(len(new_series)):
		if i in ranges:
			new_series[i] = series[i]
		else:
			new_series[i] = ''
			if i == n - 1:
				new_series[i] = series[i]
	return new_series


def output_statistics(strategy, benchmark, strategy_object):
	strategy_cum = (np.cumprod(np.array(strategy) + 1)) - 1
	benchmark_cum = (np.cumprod(np.array(benchmark) + 1)) - 1
	PnL = [strategy_cum[-1] * 100, benchmark_cum[-1] * 100]
	cumulative_return = [strategy_cum[-1], benchmark_cum[-1]]
	average_return = [250 * np.mean(strategy), 250 * np.mean(benchmark)]
	standard_deviation = [np.sqrt(250) * np.std(strategy), np.sqrt(250) * np.std(benchmark)]
	Sharpe_ratio = [np.sqrt(250) * (np.mean(strategy) - np.mean(strategy_object.df['RF'])) / np.std(strategy),
					np.sqrt(250) * (np.mean(benchmark) - np.mean(strategy_object.df['RF'])) / np.std(benchmark)]
	max_DD = [maximum_10_drawdown(strategy_cum + 1), maximum_10_drawdown(benchmark_cum + 1)]
	skewness = [stats.skew(strategy), stats.skew(benchmark)]
	kurtosis = [stats.kurtosis(strategy), stats.kurtosis(benchmark)]
	VaR = [sample_para_VaR(strategy, len(strategy)), sample_para_VaR(benchmark, len(benchmark))]

	out_df = pd.DataFrame([PnL, cumulative_return, average_return, standard_deviation, Sharpe_ratio,
						   max_DD, skewness, kurtosis, VaR])
	out_df.columns = ['strategy', 'benchmark']
	out_df.index = ['PnL', 'cumulative_return', 'average_return', 'standard_deviation',
					'Sharpe_ratio', 'max_DD', 'skewness', 'kurtosis', 'VaR']

	return out_df


if __name__ == "__main__":
	"""At the very beginning of this project, which is right after the import block,
	   we set up look back periods for Sigma and rho ( line 16 )
	   In the coming 3 lines, we set up the target period as well as our target beta
	   Then run the whole script
	   we'll get a plot together with a data frame describing portfolio's performance"""

	period1_start = "2007-4-01"
	period1_end = "2008-03-31"
	target_beta = 1
	test_strategy = Strategy(tickers, period1_start, period1_end)
	test_strategy.find_para_lookback(preset_cov_period, preset_rho_period)

	SPY = pdr.get_data_yahoo('SPY', start=period1_start, end=period1_end, interval='1d').dropna()['Adj Close']
	SPY_return = SPY.pct_change(1).dropna()
	SPY_return_cum = (np.cumprod(np.array(SPY_return) + 1)) - 1

	aaa = test_strategy.test(30, target_beta)
	cum = (np.cumprod(np.array(aaa) + 1)) - 1
	plt.plot(test_strategy.df.index[1:], SPY_return_cum, color="indianred", label='Benchmark')
	plt.plot(test_strategy.df.index[1:-3], cum, color="darkseagreen", label='Strategy Performance')
	plt.legend()
	plt.title("Target beta " + str(target_beta) + ', covariance period ' +
			  str(preset_cov_period) + ', pho period ' + str(preset_rho_period))
	pl = plt.gca()
	pl.xaxis.set_major_locator(ticker.MultipleLocator(60))
	plt.show()
	plt.close()

	result_150_90_1_p3 = output_statistics(aaa, SPY_return, test_strategy)
	print(output_statistics(aaa, SPY_return, test_strategy))

	out_df = pd.DataFrame(data=aaa, index=test_strategy.df.index[1:-3])
	out_df.to_csv("data/" + str(preset_cov_period) + '_' + str(preset_rho_period) +
				  '_' + str(target_beta) + '_P1.csv')

	# plt.figure()
	# j = 1
	# for i in ['HML', 'SMB', 'Mkt-RF']:
	# 	plt.subplot(3, 1, j)
	# 	plt.plot(FF_use[i], color='darkseagreen')
	# 	plt.title(i)
	# 	print('P_value of ADF test on ' + i + ' is '+ str(ADF(FF_use[i]).pvalue))
	# 	j += 1
	# plt.show()

	returns = pd.read_csv("data\90_150_1_P1.csv", index_col=0)
	plt.subplot(2, 1, 1)
	# 	plt.plot(FF_use[i], color='darkseagreen')
	n, bins, patches = plt.hist(SPY_return, bins=20, facecolor="lightpink",
								edgecolor="indianred", alpha=0.7)
	plt.title('SPY return')
	plt.subplot(2, 1, 2)
	plt.hist(returns, bins=20, facecolor="darkseagreen", edgecolor="darkgreen", alpha=0.7)
	plt.title('strategy return')
	plt.show()



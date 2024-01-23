import os
import sys
import logging
import datetime
import numpy as np
import pandas as pd

sys.path.append("Price_Collector")
from Portfolio import Portfolio

sys.path.append("Analysis")
from Simple_Analysis import Simple_Analysis

from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split


class Pairs_Trading_Portfolio(Portfolio):
	"""docstring for Pairs_Trading_Portfolio"""
	def __init__(self, **arg):
		super().__init__( **arg )

		dataset = self.inst_price
		train, test = train_test_split(dataset, test_size=0.6, shuffle = False)

		gama, mu  = self.make_regression(train)

		training_residue = self.get_residue(train, gama, mu)

		test_residue = self.get_residue(test, gama, mu)


		# dftest = adfuller(test_residue)
		# print(self.holdings)
		# print(dftest)


		self.inst_price = test
		self.inst_price['residue'] = test_residue
		self.inst_price['mean'] = np.mean(test_residue)
		self.inst_price['-1_std'] = np.mean(test_residue) - np.std(test_residue)
		self.inst_price['+1_std'] = np.mean(test_residue) + np.std(test_residue)		



		residue_df = pd.concat( [pd.DataFrame({'train_residue' : training_residue}), pd.DataFrame({'test_residue':test_residue}) ], axis = 1)
		ax, plot = Simple_Analysis.return_histogram(residue_df , "%s_%s.png"%(self.holdings[0],self.holdings[1]))

		self.inst_price = self.inst_price[['date','residue','mean','-1_std','+1_std']].set_index('date')
		Simple_Analysis.plot_time_series_df(self.inst_price.tail(750), linestyles = ['-','-','-.','-.'], colors =  ['w','c','c','c'] , output_path = os.path.join('data','output',"%s_%s.png"%(self.holdings[0],self.holdings[1])))



		# dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value(%)','#Lags Used','Number of Observations Used'])		
		# for key,value in dftest[4].items():

		# 	dfoutput['Critical Value (%s)' % key] = value
		# 	print(dfoutput)


	
	def make_regression(self, dataset):
		#to make a LS regression to get gama (hedge ratio) and mu (offset term)
		x, y = dataset['%s_c'%(self.holdings[0]) ].to_numpy(), dataset[ '%s_c'%(self.holdings[1])  ].to_numpy()
		gama, mu = np.linalg.lstsq(np.vstack([x, np.ones(len(x)) ]).T, y, rcond=None)[0] # slop : gama, interception : mu
		return gama, mu


	def get_residue(self, dataset, gama, mu):
		#get residue according to previous 
		x, y = dataset['%s_c'%(self.holdings[0]) ].to_numpy(), dataset[ '%s_c'%(self.holdings[1])  ].to_numpy()
		return y - x * gama + mu

	def plot_histogram():


		return 0

if __name__ == '__main__':

	# Pairs_Trading_Portfolio(holdings = ['QQQ','QQQM'], price_starting_date = '2015-01-01')
	# Pairs_Trading_Portfolio(holdings = ['SQQQ','TQQQ'], price_starting_date = '2015-01-01')
	# Pairs_Trading_Portfolio(holdings = ["0005.HK","0011.HK"], price_starting_date = '2015-01-01')		

	# Pairs_Trading_Portfolio(holdings = ['USD','HKD'], price_starting_date = '2015-01-01')
	# Pairs_Trading_Portfolio(holdings = ['EWH','EWZ'], price_starting_date = '2015-01-01')
	# Pairs_Trading_Portfolio(holdings = ['AME','DOV'], price_starting_date = '2015-01-01')
	Pairs_Trading_Portfolio(holdings = ["VIXY","VXX"], price_starting_date = '2015-01-01')

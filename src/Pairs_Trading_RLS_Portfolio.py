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


class Pairs_Trading_RLS_Portfolio(Portfolio):
	"""docstring for Pairs_Trading_Portfolio"""
	def __init__(self, **arg):
		super().__init__( **arg )

		dataset = self.inst_price

		#split according to sliding windows
		dataset =  Portfolio.sliding_window_split(dataset, 10)

		# dataset =  Portfolio.sliding_window_split(dataset, 25)

		train, test = train_test_split(dataset, test_size=0.6, shuffle = False)

		training_residue, test_residue = [], []		
		for item in train:
			regression_input = item.iloc[:-1]
			gama, mu = self.make_regression(regression_input)
			residue = self.get_residue(item, gama, mu)
			training_residue +=  [residue[0] ]


		for item in test:
			regression_input = item.iloc[:-1]
			gama, mu = self.make_regression(regression_input)
			residue = self.get_residue(item, gama, mu)
			test_residue += [ residue[-1] ]


		self.test_dftest_p_value , self.train_dftest_p_value = adfuller(test_residue)[1] , adfuller(training_residue)[1]
		

		print("\n %s train P-value : %.5f%% ; test P-value : %.5f%%\n"%(self.holdings, self.train_dftest_p_value*100 , self.test_dftest_p_value*100 ) )


		residue_df = pd.concat( [pd.DataFrame({'train_residue' : training_residue}), pd.DataFrame({'test_residue':test_residue}) ], axis = 1)
		Simple_Analysis.return_histogram(residue_df , os.path.join('data','output_RLS',"%s_%s_histogram.png"%(self.holdings[0], self.holdings[1])) )
		
		Simple_Analysis.plot_adf(residue_df, os.path.join('data','output_RLS',"%s_%s_adf.png"%(self.holdings[0],self.holdings[1])) )

		train_residue_df = pd.DataFrame({ 'residue' : training_residue, 'mean' : np.mean(training_residue), '-1_std' : np.mean(training_residue) - np.std(training_residue), '+1_std' : np.mean(training_residue) + np.std(training_residue) }, index = [ item.index.values[0] for item in train ] )
		Simple_Analysis.plot_time_series_df(train_residue_df, linestyles = ['dotted','-','-.','-.'], colors =  ['w','c','c','c'] , output_path = os.path.join('data','output_RLS',"%s_%s_train.png"%(self.holdings[0],self.holdings[1])))


		test_residue_df = pd.DataFrame({ 'residue' : test_residue, 'mean' : np.mean(test_residue), '-1_std' : np.mean(test_residue) - np.std(test_residue), '+1_std' : np.mean(test_residue) +  np.std(test_residue) }, index = [ item.index.values[-1] for item in test ] )
		Simple_Analysis.plot_time_series_df(test_residue_df, linestyles = ['dotted','-','-.','-.'], colors =  ['w','c','c','c'] , output_path = os.path.join('data','output_RLS',"%s_%s_test.png"%(self.holdings[0],self.holdings[1])))

		

if __name__ == '__main__':

	# Pairs_Trading_RLS_Portfolio(holdings = ['USD','HKD'], price_starting_date = '2015-01-01')
	Pairs_Trading_RLS_Portfolio(holdings = ["0005.HK","0011.HK"], price_starting_date = '2015-01-01')		
	Pairs_Trading_RLS_Portfolio(holdings = ['EWH','EWZ'], price_starting_date = '2015-01-01')
	Pairs_Trading_RLS_Portfolio(holdings = ['AME','DOV'], price_starting_date = '2015-01-01')

	Pairs_Trading_RLS_Portfolio(holdings = ['SQQQ','TQQQ'], price_starting_date = '2015-01-01')
	Pairs_Trading_RLS_Portfolio(holdings = ['QQQ','QQQM'], price_starting_date = '2015-01-01')
	Pairs_Trading_RLS_Portfolio(holdings = ["VIXY","VXX"], price_starting_date = '2015-01-01')

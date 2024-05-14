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

sys.path.append("src")
from Signal_Generator import Signal_Generator

class Pairs_Trading_RLS_Portfolio(Portfolio):
	"""docstring for Pairs_Trading_RLS_Portfolio

	NOTE:

	Residules of RLS Regression are self correlated show in adf plot.
	To fix, proposed MAAR adding residule as an input (TBD), and Kalman_filter regression

	"""
	def __init__(self, **arg):
		super().__init__( **arg )

		dataset = self.inst_price
		window_length = 60
		#split according to sliding windows
		dataset =  Portfolio.sliding_window_split(dataset, window_length)

		
		residue_list, mu_list, gama_list = [], [], { hold : [] for hold in self.holdings}
		records = []
		for item in dataset:
			records += [ item.iloc[[-1]].copy() ]
			regression_input = item.iloc[:-1]
			gama, mu = self.make_regression(regression_input[ [ "%s_c"%(hold) for hold in self.holdings]  ] )

			residue = self.get_residue(item[ [ "%s_c"%(hold) for hold in self.holdings]  ] , gama, mu)
			

			residue_list += [residue[-1] ]			
			
			mu_list += [ mu[0] ]

			for i in range(0,len(self.holdings)):
				if i == 0:
					gama_list[ self.holdings[i] ] += [1]
				else:
					gama_list[ self.holdings[i] ] += [gama[i-1]]

		self.dataset = pd.concat(records)

		for hold in self.holdings:
			self.dataset['%s_gama'%(hold)] = gama_list[hold]

		self.dataset['mu'] = mu_list
		self.dataset['residue'] = residue_list
		

		self.test_dftest_p_value , self.train_dftest_p_value = adfuller( self.dataset[ self.dataset["dataset_tag"] == "test" ]['residue'] )[1] , adfuller(self.dataset[ self.dataset["dataset_tag"] == "train" ]['residue'])[1]
		

		print("\n %s\
		\ntrain P-value : %.3g%% residue mean %.3f ,std %.3f ; \
		\ntest  P-value : %.3g%% residue mean %.3f ,std %.3f \n"%(
		self.holdings, self.train_dftest_p_value*100 , np.mean(  self.dataset[ self.dataset['dataset_tag'] == 'train' ]['residue'] ), np.std( self.dataset[ self.dataset['dataset_tag'] == 'train' ]['residue'] ) , 
		self.test_dftest_p_value*100, np.mean(  self.dataset[ self.dataset['dataset_tag'] == 'test' ]['residue'] ), np.std( self.dataset[ self.dataset['dataset_tag'] == 'test' ]['residue'] )  ) )


		sg = Signal_Generator(self.dataset, self.holdings)
		sg.generate_optimal()
		self.dataset = sg.dataset

	def plot(self):

		# train, test = train_test_split(self.dataset, test_size=0.7, shuffle = False)
		train, test = self.dataset[ self.dataset['dataset_tag'] == 'train'  ], self.dataset[ self.dataset['dataset_tag'] == 'test'  ]

		residue_df = pd.concat( [pd.DataFrame({'train_residue' : train['residue']}), pd.DataFrame({'test_residue': test['residue'] }) ], axis = 1)
		Simple_Analysis.return_histogram(residue_df , os.path.join('data','output_RLS',"%s_%s_histogram.png"%(self.holdings[0], self.holdings[1])) )
		
		Simple_Analysis.plot_adf(residue_df, os.path.join('data','output_RLS',"%s_%s_adf.png"%(self.holdings[0],self.holdings[1])) )

		test_residue_df = pd.DataFrame({ 'residue' : test['residue'], 'mean' : np.mean(test['residue']), '-1_std' : np.mean(test['residue']) - np.std(test['residue']), '+1_std' : np.mean(test['residue']) +  np.std(test['residue']) }, index = test.index)
		Simple_Analysis.plot_time_series_df(test_residue_df, linestyles = ['dotted','-','-.','-.'], colors =  ['w','c','c','c'] , output_path = os.path.join('data','output_RLS',"%s_%s_test.png"%(self.holdings[0],self.holdings[1])))
		
		train_residue_df = pd.DataFrame({ 'residue' : train['residue'], 'mean' : np.mean(train['residue']), '-1_std' : np.mean(train['residue']) - np.std(train['residue']), '+1_std' : np.mean(train['residue']) + np.std(train['residue']) }, index = train.index)
		Simple_Analysis.plot_time_series_df(train_residue_df, linestyles = ['dotted','-','-.','-.'], colors =  ['w','c','c','c'] , output_path = os.path.join('data','output_RLS',"%s_%s_train.png"%(self.holdings[0],self.holdings[1])))



if __name__ == '__main__':

	# Pairs_Trading_RLS_Portfolio(holdings = ['USD','HKD'], price_starting_date = '2015-01-01')
	RLS = Pairs_Trading_RLS_Portfolio(holdings = ["PEP","KO"], price_starting_date = '2015-01-01')
	# RLS.plot()
	RLS.dataset.to_csv("./data/output_RLS/RLS_PEP_KO.csv")

	RLS = Pairs_Trading_RLS_Portfolio(holdings = ["VOO","SPY"], price_starting_date = '2015-01-01')
	# RLS.plot()
	RLS.dataset.to_csv("./data/output_RLS/RLS_VOO_SPY.csv")

	RLS = Pairs_Trading_RLS_Portfolio(holdings = ['QQQ','QQQM'], price_starting_date = '2015-01-01')
	# RLS.plot()
	RLS.dataset.to_csv("./data/output_RLS/RLS_QQQ_QQQM.csv")

	RLS = Pairs_Trading_RLS_Portfolio(holdings = ["0005.HK","0011.HK"], price_starting_date = '2015-01-01')	
	# RLS.plot()
	RLS.dataset.to_csv("./data/output_RLS/RLS_0005_0011.csv")

	# Pairs_Trading_RLS_Portfolio(holdings = ['EWH','EWZ'], price_starting_date = '2015-01-01')
	# Pairs_Trading_RLS_Portfolio(holdings = ['AME','DOV'], price_starting_date = '2015-01-01')

	# Pairs_Trading_RLS_Portfolio(holdings = ['SQQQ','TQQQ'], price_starting_date = '2015-01-01')
	# Pairs_Trading_RLS_Portfolio(holdings = ["VIXY","VXX"], price_starting_date = '2015-01-01')

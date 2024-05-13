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


class Pairs_Trading_MAAR_Portfolio(Portfolio):
	"""docstring for Pairs_Trading_Portfolio"""
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
		

		print("\n %s train P-value :  %.3g%% ; test P-value : %.3g%%\n"%(self.holdings, self.train_dftest_p_value*100 , self.test_dftest_p_value*100 ) )


		# residue_df = pd.concat( [pd.DataFrame({'train_residue' : training_residue}), pd.DataFrame({'test_residue':test_residue}) ], axis = 1)
		# Simple_Analysis.return_histogram(residue_df , os.path.join('data','output_RLS',"%s_%s_histogram.png"%(self.holdings[0], self.holdings[1])) )
		
		# Simple_Analysis.plot_adf(residue_df, os.path.join('data','output_RLS',"%s_%s_adf.png"%(self.holdings[0],self.holdings[1])) )

		# train_residue_df = pd.DataFrame({ 'residue' : training_residue, 'mean' : np.mean(training_residue), '-1_std' : np.mean(training_residue) - np.std(training_residue), '+1_std' : np.mean(training_residue) + np.std(training_residue) }, index = [ item.index.values[0] for item in train ] )
		# Simple_Analysis.plot_time_series_df(train_residue_df, linestyles = ['dotted','-','-.','-.'], colors =  ['w','c','c','c'] , output_path = os.path.join('data','output_RLS',"%s_%s_train.png"%(self.holdings[0],self.holdings[1])))


		# test_residue_df = pd.DataFrame({ 'residue' : test_residue, 'mean' : np.mean(test_residue), '-1_std' : np.mean(test_residue) - np.std(test_residue), '+1_std' : np.mean(test_residue) +  np.std(test_residue) }, index = [ item.index.values[-1] for item in test ] )
		# Simple_Analysis.plot_time_series_df(test_residue_df, linestyles = ['dotted','-','-.','-.'], colors =  ['w','c','c','c'] , output_path = os.path.join('data','output_RLS',"%s_%s_test.png"%(self.holdings[0],self.holdings[1])))

		

if __name__ == '__main__':

	# Pairs_Trading_MAAR_Portfolio(holdings = ['SQQQ','TQQQ'], price_starting_date = '2015-01-01')
	# Pairs_Trading_MAAR_Portfolio(holdings = ['USD','HKD'], price_starting_date = '2015-01-01')
	# Pairs_Trading_MAAR_Portfolio(holdings = ["0005.HK","0011.HK"], price_starting_date = '2015-01-01')		
	# Pairs_Trading_MAAR_Portfolio(holdings = ['EWH','EWZ'], price_starting_date = '2015-01-01')
	# Pairs_Trading_MAAR_Portfolio(holdings = ['AME','DOV'], price_starting_date = '2015-01-01')


	pep_ko = Pairs_Trading_MAAR_Portfolio(holdings = ["PEP","KO"], price_starting_date = '2015-01-01')	
	pep_ko.dataset.to_csv("./data/output_MAAR/MAAR_PEP_KO.csv")

	QQQ_QQQM = Pairs_Trading_MAAR_Portfolio(holdings = ['QQQ','QQQM'], price_starting_date = '2015-01-01')
	QQQ_QQQM.dataset.to_csv("./data/output_MAAR/MAAR_QQQ_QQQM.csv")

	VOO_SPY = Pairs_Trading_MAAR_Portfolio(holdings = ['VOO','SPY'], price_starting_date = '2015-01-01')
	VOO_SPY.dataset.to_csv("./data/output_MAAR/MAAR_VOO_SPY.csv")

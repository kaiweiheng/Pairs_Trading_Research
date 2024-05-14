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

class Pairs_Trading_Portfolio(Portfolio):
	"""docstring for Pairs_Trading_Portfolio"""
	def __init__(self, **arg):
		super().__init__( **arg )

		dataset = self.inst_price
		train, test = self.inst_price[ self.inst_price['dataset_tag'] == 'train'], self.inst_price[ self.inst_price['dataset_tag'] == 'test']
		# train, test = train_test_split(dataset, test_size=0.7, shuffle = False)
		
		gama, mu  = self.make_regression(train[ [ "%s_c"%(hold) for hold in self.holdings]  ])

		training_residue = self.get_residue(train[ [ "%s_c"%(hold) for hold in self.holdings]  ], gama, mu)
		
		test_residue = self.get_residue(test[ [ "%s_c"%(hold) for hold in self.holdings]  ], gama, mu)
		
		self.test_dftest_p_value , self.train_dftest_p_value = adfuller(test_residue)[1] , adfuller(training_residue)[1]
	
		print("\n %s train P-value : %.3g%% ; test P-value : %.3g%% \n"%(self.holdings, self.train_dftest_p_value*100 , self.test_dftest_p_value*100 ) )


		dataset['%s_gama'%(self.holdings[0])] = [ 1 for _ in range(0,len(dataset)) ]

		for i in range(1 , len(self.holdings)):
			dataset['%s_gama'%(self.holdings[i])] = [ gama[i-1] for _ in range(0,len(dataset)) ]

		dataset['mu'] = [mu[0] for i in range(0, len(dataset))]

		residue = training_residue.tolist() 
		residue += test_residue.tolist() 
		dataset['residue'] = residue
		
		self.dataset = dataset

		sg = Signal_Generator(self.dataset, self.holdings)
		sg.generate_optimal()
		self.dataset = sg.dataset		

	def plot(self):

		# train, test = train_test_split(self.dataset, test_size=0.7, shuffle = False)
		train, test = self.dataset[ self.dataset['dataset_tag'] == 'train'  ], self.dataset[ self.dataset['dataset_tag'] == 'test'  ]

		residue_df = pd.concat( [pd.DataFrame({'train_residue' : train['residue']}), pd.DataFrame({'test_residue': test['residue'] }) ], axis = 1)
		Simple_Analysis.return_histogram(residue_df , os.path.join('data','output',"%s_%s_histogram.png"%(self.holdings[0], self.holdings[1])) )
		
		Simple_Analysis.plot_adf(residue_df, os.path.join('data','output',"%s_%s_adf.png"%(self.holdings[0],self.holdings[1])) )

		train_residue_df = pd.DataFrame({ 'residue' : train['residue'], 'mean' : np.mean(train['residue']), '-1_std' : np.mean(train['residue']) - np.std(train['residue']), '+1_std' : np.mean(train['residue']) + np.std(train['residue']) }, index = train.index)
		Simple_Analysis.plot_time_series_df(train_residue_df, linestyles = ['dotted','-','-.','-.'], colors =  ['w','c','c','c'] , output_path = os.path.join('data','output',"%s_%s_train.png"%(self.holdings[0],self.holdings[1])))


		test_residue_df = pd.DataFrame({ 'residue' : test['residue'], 'mean' : np.mean(test['residue']), '-1_std' : np.mean(test['residue']) - np.std(test['residue']), '+1_std' : np.mean(test['residue']) +  np.std(test['residue']) }, index = test.index)
		Simple_Analysis.plot_time_series_df(test_residue_df, linestyles = ['dotted','-','-.','-.'], colors =  ['w','c','c','c'] , output_path = os.path.join('data','output',"%s_%s_test.png"%(self.holdings[0],self.holdings[1])))


if __name__ == '__main__':


	pep_ko = Pairs_Trading_Portfolio(holdings = ["PEP","KO"], price_starting_date = '2015-01-01')
	pep_ko.plot()
	pep_ko.dataset.to_csv("./data/output/PEP_KO.csv")

	QQQ_QQQM = Pairs_Trading_Portfolio(holdings = ['QQQ','QQQM'], price_starting_date = '2015-01-01')
	QQQ_QQQM.dataset.to_csv("./data/output/QQQ_QQQM.csv")

	VOO_SPY = Pairs_Trading_Portfolio(holdings = ['VOO','SPY'], price_starting_date = '2015-01-01')
	VOO_SPY.dataset.to_csv("./data/output/VOO_SPY.csv")
	# a = Pairs_Trading_Portfolio(holdings = ["0005.HK","0011.HK"], price_starting_date = '2015-01-01')		
	# Pairs_Trading_Portfolio(holdings = ['EWH','EWZ'], price_starting_date = '2015-01-01')
	# Pairs_Trading_Portfolio(holdings = ['AME','DOV'], price_starting_date = '2015-01-01')
	# Pairs_Trading_Portfolio(holdings = ['SQQQ','TQQQ'], price_starting_date = '2015-01-01')
	# Pairs_Trading_Portfolio(holdings = ["VIXY","VXX"], price_starting_date = '2015-01-01')

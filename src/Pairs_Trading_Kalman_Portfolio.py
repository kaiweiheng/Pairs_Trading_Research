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
from KalmanFilter import KalmanFilter


class Pairs_Trading_Kalman_Portfolio(Portfolio):
	"""docstring for Pairs_Trading_Portfolio"""
	def __init__(self, **arg):
		super().__init__( **arg )

		dataset_raw = self.inst_price
		calibrate_length = 50

		#split according to sliding windows
		dataset =  Portfolio.sliding_window_split(dataset_raw, 1)


		#init KF
		regression_input = dataset[0]
		y, x = Portfolio.split_variables_explanatory_and_independent(regression_input[ [ "%s_c"%(hold) for hold in self.holdings]  ], self.holdings)		
		x = np.hstack([x, np.ones( (len(x), 1)  ) ])
		F = np.array([[1, 0],[0, 1]])
		R = np.array([0.1]).reshape(1, 1) #estimation of noise covariance Q and R to be implemented
		kf = KalmanFilter(F = F, H = x, R = R)


		# calibration KalmanFilter
		for item in dataset[:calibrate_length]:
			y, x = Portfolio.split_variables_explanatory_and_independent(item[ [ "%s_c"%(hold) for hold in self.holdings]  ], self.holdings)
			x = np.hstack([x, np.ones( (len(x), 1)  ) ])
			r_test =  y - np.dot(x , kf.predict() )

			kf.H = x
			kf.update(y)

			_ =  y - np.dot(x , kf.x )
		
		dataset = dataset[calibrate_length:]


		residue_list, mu_list, gama_list = [], [], { hold : [] for hold in self.holdings}
		records = []

		for item in dataset:
			records += [ item.iloc[[-1]].copy() ]
			y, x = Portfolio.split_variables_explanatory_and_independent(item[ [ "%s_c"%(hold) for hold in self.holdings]  ], self.holdings)

			x = np.hstack([x, np.ones( (len(x), 1)  ) ])
			residue =  y - np.dot(x , kf.predict() )

			kf.H = x

			# residue above equivalent to below
			# residue = y - np.dot(x , kf.x )

			residue_list += [residue.tolist()[0][0]]
			mu_list += [kf.x[-1][0]]

			for i in range(0,len(self.holdings)):
				if i == 0:
					gama_list[ self.holdings[i] ] += [1]
				else:
					gama_list[ self.holdings[i] ] += [ kf.x[i-1][0]]


			kf.update(y)

			# training_residue +=  [residue.tolist()[0][0]]
			# gamas += [kf.x[0]]
			# mus += [kf.x[1]]
			

		# self.dataset = self.inst_price.iloc[calibrate_length + 1:].copy()
		self.dataset = pd.concat(records)
	
		for hold in self.holdings:
			self.dataset['%s_gama'%(hold)] = gama_list[hold]

		self.dataset['mu'] = mu_list
		self.dataset['residue'] = residue_list



		self.test_dftest_p_value , self.train_dftest_p_value = adfuller( self.dataset[ self.dataset['dataset_tag'] == 'test' ]['residue'] )[1] , adfuller( self.dataset[ self.dataset['dataset_tag'] == 'train' ]['residue'] )[1]
		

		print("\n %s train P-value : %.3g%% ; test P-value : %.3g%% \n"%(self.holdings, self.train_dftest_p_value*100 , self.test_dftest_p_value*100 ) )



	def plot(self):

		train, test = train_test_split(self.dataset, test_size=0.7, shuffle = False)

		# print(pd.DataFrame({'test_residue': test['residue'] }).tail(50))
		# print(pd.DataFrame({'test_residue': test['gama'] }).tail(10))

		residue_df = pd.concat( [pd.DataFrame({'train_residue' : train['residue']}), pd.DataFrame({'test_residue': test['residue'] }) ], axis = 1)
		Simple_Analysis.return_histogram(residue_df , os.path.join('data','output_Kalman',"%s_%s_histogram.png"%(self.holdings[0], self.holdings[1])) )
		
		Simple_Analysis.plot_adf(residue_df, os.path.join('data','output_Kalman',"%s_%s_adf.png"%(self.holdings[0],self.holdings[1])) )

		train_residue_df = pd.DataFrame({ 'residue' : train['residue'], 'mean' : np.mean(train['residue']), '-1_std' : np.mean(train['residue']) - np.std(train['residue']), '+1_std' : np.mean(train['residue']) + np.std(train['residue']) }, index = train.index)
		Simple_Analysis.plot_time_series_df(train_residue_df, linestyles = ['dotted','-','-.','-.'], colors =  ['w','c','c','c'] , output_path = os.path.join('data','output_Kalman',"%s_%s_train.png"%(self.holdings[0],self.holdings[1])))


		test_residue_df = pd.DataFrame({ 'residue' : test['residue'], 'mean' : np.mean(test['residue']), '-1_std' : np.mean(test['residue']) - np.std(test['residue']), '+1_std' : np.mean(test['residue']) +  np.std(test['residue']) }, index = test.index)
		Simple_Analysis.plot_time_series_df(test_residue_df, linestyles = ['dotted','-','-.','-.'], colors =  ['w','c','c','c'] , output_path = os.path.join('data','output_Kalman',"%s_%s_test.png"%(self.holdings[0],self.holdings[1])))


if __name__ == '__main__':

	KF = Pairs_Trading_Kalman_Portfolio(holdings = ['PEP','KO'], price_starting_date = '2015-01-01')
	KF.plot()
	KF.dataset.to_csv("./data/output_Kalman/Kalman_PEP_KO.csv")

	KF = Pairs_Trading_Kalman_Portfolio(holdings = ['VOO','SPY'], price_starting_date = '2015-01-01')
	KF.plot()
	KF.dataset.to_csv("./data/output_Kalman/Kalman_VOO_SPY.csv")

	KF = Pairs_Trading_Kalman_Portfolio(holdings = ['QQQ','QQQM'], price_starting_date = '2015-01-01')
	KF.plot()
	KF.dataset.to_csv("./data/output_Kalman/Kalman_QQQ_QQQM.csv")	

	# Pairs_Trading_Kalman_Portfolio(holdings = ['SQQQ','TQQQ'], price_starting_date = '2015-01-01').plot()
	# Pairs_Trading_Kalman_Portfolio(holdings = ["0005.HK","0011.HK"], price_starting_date = '2015-01-01').plot()	
	# Pairs_Trading_Kalman_Portfolio(holdings = ['EWH','EWZ'], price_starting_date = '2015-01-01').plot()
	# Pairs_Trading_Kalman_Portfolio(holdings = ['AME','DOV'], price_starting_date = '2015-01-01').plot()

	# Pairs_Trading_Kalman_Portfolio(holdings = ['QQQ','QQQM'], price_starting_date = '2015-01-01').plot()
	# Pairs_Trading_Kalman_Portfolio(holdings = ["VIXY","VXX"], price_starting_date = '2015-01-01').plot()

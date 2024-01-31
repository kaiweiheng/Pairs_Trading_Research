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

		dataset = self.inst_price

		#split according to sliding windows
		dataset =  Portfolio.sliding_window_split(dataset, 1)

		# dataset =  Portfolio.sliding_window_split(dataset, 25)

		train, test = train_test_split(dataset, test_size=0.7, shuffle = False)

		regression_input = train[0]
		y, x = Portfolio.split_variables_explanatory_and_independent(regression_input, self.holdings)		
		x = np.hstack([x, np.ones( (len(x), 1)  ) ])
		F = np.array([[1, 0],[0, 1]])
		R = np.array([0.1]).reshape(1, 1) #estimation of noise covariance Q and R to be implemented
		kf = KalmanFilter(F = F, H = x, R = R)


		# calibration KalmanFilter
		for item in train[:20]:
			y, x = Portfolio.split_variables_explanatory_and_independent(item, self.holdings)
			x = np.hstack([x, np.ones( (len(x), 1)  ) ])
			r_test =  y - np.dot(x , kf.predict() )

			kf.H = x
			kf.update(y)

			r_train =  y - np.dot(x , kf.x )

		train = train[20:]
		training_residue, test_residue = [], []		
		for item in train:
			y, x = Portfolio.split_variables_explanatory_and_independent(item, self.holdings)
			x = np.hstack([x, np.ones( (len(x), 1)  ) ])
			r_test =  y - np.dot(x , kf.predict() )

			kf.H = x
			kf.update(y)

			r_train = y - np.dot(x , kf.x )
			training_residue +=  [r_train.tolist()[0][0]]

		flag = False
		for item in test:
			y, x = Portfolio.split_variables_explanatory_and_independent(item, self.holdings)
			x = np.hstack([x, np.ones( (len(x), 1)  ) ])
			r_test = y - np.dot(x , kf.predict() )

			if flag:
				print(r_test.tolist()[0][0])
				print(item)				
				flag = False

			if r_test.tolist()[0][0] > 2 or r_test.tolist()[0][0] < -2:
				print(r_test.tolist()[0][0])
				print(kf.x)
				print(item)
				print("  ")
				flag = True


			

			test_residue +=  [r_test.tolist()[0][0] ]

			kf.H = x
			kf.update(y)

			r_train  =  y - np.dot(x , kf.x )


		self.test_dftest_p_value , self.train_dftest_p_value = adfuller(test_residue)[1] , adfuller(training_residue)[1]
		

		print("\n %s train P-value : %.5f%% ; test P-value : %.5f%%\n"%(self.holdings, self.train_dftest_p_value*100 , self.test_dftest_p_value*100 ) )


		residue_df = pd.concat( [pd.DataFrame({'train_residue' : training_residue}), pd.DataFrame({'test_residue':test_residue}) ], axis = 1)
		Simple_Analysis.return_histogram(residue_df , os.path.join('data','output_Kalman',"%s_%s_histogram.png"%(self.holdings[0], self.holdings[1])) )
		
		Simple_Analysis.plot_adf(residue_df, os.path.join('data','output_Kalman',"%s_%s_adf.png"%(self.holdings[0],self.holdings[1])) )

		train_residue_df = pd.DataFrame({ 'residue' : training_residue, 'mean' : np.mean(training_residue), '-1_std' : np.mean(training_residue) - np.std(training_residue), '+1_std' : np.mean(training_residue) + np.std(training_residue) }, index = [ item.index.values[0] for item in train ] )
		Simple_Analysis.plot_time_series_df(train_residue_df, linestyles = ['dotted','-','-.','-.'], colors =  ['w','c','c','c'] , output_path = os.path.join('data','output_Kalman',"%s_%s_train.png"%(self.holdings[0],self.holdings[1])))


		test_residue_df = pd.DataFrame({ 'residue' : test_residue, 'mean' : np.mean(test_residue), '-1_std' : np.mean(test_residue) - np.std(test_residue), '+1_std' : np.mean(test_residue) +  np.std(test_residue) }, index = [ item.index.values[-1] for item in test ] )
		Simple_Analysis.plot_time_series_df(test_residue_df, linestyles = ['dotted','-','-.','-.'], colors =  ['w','c','c','c'] , output_path = os.path.join('data','output_Kalman',"%s_%s_test.png"%(self.holdings[0],self.holdings[1])))

		
	def kf_prediction():

		return 0

	def kf_updated():

		return 0

if __name__ == '__main__':

	# Pairs_Trading_Kalman_Portfolio(holdings = ['USD','HKD'], price_starting_date = '2015-01-01')
	Pairs_Trading_Kalman_Portfolio(holdings = ['SQQQ','TQQQ'], price_starting_date = '2015-01-01')
	# Pairs_Trading_Kalman_Portfolio(holdings = ["0005.HK","0011.HK"], price_starting_date = '2015-01-01')		
	# Pairs_Trading_Kalman_Portfolio(holdings = ['EWH','EWZ'], price_starting_date = '2015-01-01')
	# Pairs_Trading_Kalman_Portfolio(holdings = ['AME','DOV'], price_starting_date = '2015-01-01')

	# Pairs_Trading_Kalman_Portfolio(holdings = ['QQQ','QQQM'], price_starting_date = '2015-01-01')
	Pairs_Trading_Kalman_Portfolio(holdings = ["VIXY","VXX"], price_starting_date = '2015-01-01')

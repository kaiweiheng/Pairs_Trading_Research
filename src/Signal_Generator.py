import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from Pairs_Trading_Portfolio import Pairs_Trading_Portfolio 
# from Pairs_Trading_RLS_Portfolio import Pairs_Trading_RLS_Portfolio
# from Pairs_Trading_Kalman_Portfolio import Pairs_Trading_Kalman_Portfolio

class Signal_Generator(object):
	"""docstring for Signal_Generator"""
	def __init__(self, residue_df, holdings):
		self.holdings = holdings
		self.dataset = residue_df
		self.train_dataset, self.test_dataset = self.dataset[ self.dataset['dataset_tag'] == 'train'], self.dataset[ self.dataset['dataset_tag'] == 'test']

	
	def generate_by_N_std(self, N_std = 2.5):
		self.train_residue_mean, self.train_residue_std = self.train_dataset['residue'].mean() , self.train_dataset['residue'].std()

		self.upper_bound, self.lower_bound =  self.train_residue_mean + N_std * self.train_residue_std , self.train_residue_mean - N_std * self.train_residue_std
		self.make_signal()

	def generate_by_quantile(self, quantile = 0.05):
		self.upper_bound, self.lower_bound = self.train_dataset['residue'].quantile(1-quantile/2), self.train_dataset['residue'].quantile(quantile/2)
		self.make_signal()
		


	def generate_optimal(self): #now's parameteric
		# TBD Non-parametric approach∗
		#convert to cdf
		train_residue_mean = self.train_dataset['residue'].mean()
		H,bin_edges = np.histogram( self.train_dataset['residue'] , bins = int(len(self.train_dataset['residue'])/1.2),  density = True )
		dx = bin_edges[1] - bin_edges[0]
		cumulative_probabilities = np.cumsum(H)*dx
		

		bin_edges = bin_edges[1:]

		
		profit_dictionary = dict()
		# find upper_bound, lower_bound, where max(profit), profit = (edge - mean)*probability
		for i in range(0, len(bin_edges)):
			cumulative_probability, edge = cumulative_probabilities[i], bin_edges[i]

			profit_dictionary[edge] = abs(edge - train_residue_mean) * min( cumulative_probability, 1 - cumulative_probability)


		profit_dictionary = pd.DataFrame( {'edge' : profit_dictionary.keys() , 'profit' : profit_dictionary.values() } )
		
		# fig, ax = plt.subplots()
		# ax.plot(profit_dictionary['edge'], profit_dictionary['profit'])
		# fig.savefig('./data/tmp.png', dpi = 250)
		self.upper_bound = profit_dictionary.loc[ profit_dictionary['edge'] > train_residue_mean ]
		self.upper_bound = self.upper_bound.loc[  self.upper_bound['profit'] == self.upper_bound['profit'].max()  ]
		self.upper_bound = self.upper_bound['edge'].values[0]

		self.lower_bound = profit_dictionary.loc[ profit_dictionary['edge'] < train_residue_mean ]
		self.lower_bound = self.lower_bound.loc[ self.lower_bound['profit'] == self.lower_bound['profit'].max() ]
		self.lower_bound = self.lower_bound['edge'].values[0]

		self.make_signal()

	def make_signal(self):
		self.dataset['Signal'] = 0

		for hold in self.holdings:
			self.dataset["%s_Signal"%(hold)] = 0

		'''
		### for the whole portfolio level ###

		Short the Signal, Signal = -1

		Long the Signal,  Signal = 1

		#####################################
		'''
		#if residue > upper_bond, we short the portfolio
		self.dataset.loc[self.dataset['residue'] > self.upper_bound , 'Signal' ] = -1
		
		#if residue > upper_bond, we long the portfolio
		self.dataset.loc[self.dataset['residue'] < self.lower_bound , 'Signal' ] = 1

		'''
		### for each stock in the portfolio, assumeing all Gamas are positive ###

		--Regression Target (the first stock)

			Short the whole portfolio -> Short this stock   -1 ->-1

			Long the whole portfolio  -> Long this stock     1 -> 1

		--Explanatory Input (Rest of the Stock)

			Short the whole portfolio -> Long this stock   -1 -> 1

			Long the whole portfolio  -> Short this stock   1 ->-1


		######################################

		'''

		for i in range(0,len(self.holdings)):
			if i == 0:
				self.dataset.loc[  self.dataset['Signal'] == -1 , "%s_Signal"%(self.holdings[i]) ] =  1
				self.dataset.loc[  self.dataset['Signal'] ==  1 , "%s_Signal"%(self.holdings[i]) ] = -1				

			else:
				self.dataset.loc[  self.dataset['Signal'] == -1 , "%s_Signal"%(self.holdings[i]) ] = -1
				self.dataset.loc[  self.dataset['Signal'] ==  1 , "%s_Signal"%(self.holdings[i]) ] =  1				



if __name__ == '__main__':
	KF = Pairs_Trading_Kalman_Portfolio(holdings = ['PEP','KO'], price_starting_date = '2015-01-01')
	KF.plot()

	KF = Signal_Generator(residue_df = KF.dataset, holdings = ['PEP','KO'] )
	# KF.generate_by_N_std(2.5)
	# KF.generate_by_quantile(0.05)
	KF.generate_optimal()
	KF.dataset.to_csv("./data/output_Kalman/Kalman_PEP_KO.csv")

	KF = Pairs_Trading_Kalman_Portfolio(holdings = ['VOO','SPY'], price_starting_date = '2015-01-01')
	# KF.plot()
	KF = Signal_Generator(residue_df = KF.dataset, holdings = ['VOO','SPY'] )
	# KF.generate_by_N_std(2.5)
	KF.generate_optimal()
	KF.dataset.to_csv("./data/output_Kalman/Kalman_VOO_SPY.csv")

	KF = Pairs_Trading_Kalman_Portfolio(holdings = ['QQQ','QQQM'], price_starting_date = '2015-01-01')
	# KF.plot()
	KF = Signal_Generator(residue_df = KF.dataset, holdings = ['QQQ','QQQM'] )
	KF.generate_by_N_std(2.5)
	KF.dataset.to_csv("./data/output_Kalman/Kalman_QQQ_QQQM.csv")	

	KF = Pairs_Trading_Kalman_Portfolio(holdings = ["0005.HK","0011.HK"], price_starting_date = '2015-01-01')
	# KF.plot()
	KF = Signal_Generator(residue_df = KF.dataset, holdings = ["0005.HK","0011.HK"] )
	KF.generate_optimal()
	KF.dataset.to_csv("./data/output_Kalman/Kalman_0005_0011_OPT.csv")		
	KF.generate_by_N_std(2.5)
	KF.dataset.to_csv("./data/output_Kalman/Kalman_0005_0011.csv")		
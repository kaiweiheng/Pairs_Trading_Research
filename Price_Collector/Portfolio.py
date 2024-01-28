import sys
import logging
import datetime

sys.path.append("Price_Collector")
from Instrument import *

class Portfolio(object):
	"""docstring for Portfolio"""
	def __init__(self, **arg):
		super(Portfolio, self).__init__()
		self.holdings = arg['holdings']
		self.price_starting_date = arg['price_starting_date']
		self.inst_list = []

		for inst in self.holdings:
			self.inst_list.append(Instrument(instrument_code = inst, price_starting_date = self.price_starting_date))

		self.inst_price = self.inst_list[0].price
		for inst in self.inst_list[1:]:
			self.inst_price = pd.merge(self.inst_price, inst.price, how = 'inner', left_on = 'date', right_on = 'date').set_index('date')

		#do 
		# self.inst_stad_price = self.inst_list[0].stad_price
		# for inst in self.inst_list[1:]:
		# 	self.inst_stad_price = pd.merge(self.inst_stad_price, inst.stad_price, how = 'inner', left_on = 'date', right_on = 'date')			

		# self.inst_re = self.inst_list[0].price_return
		# for inst in self.inst_list[1:]:
		# 	self.inst_re = pd.merge(self.inst_re, inst.price_return, how = 'inner', left_on = 'date', right_on = 'date')

		# self.inst_stad_re = self.inst_list[0].stad_price_return
		# for inst in self.inst_list[1:]:
		# 	self.inst_stad_re = pd.merge(self.inst_stad_re, inst.stad_price_return, how = 'inner', left_on = 'date', right_on = 'date')

	def make_regression(self, dataset):
		#to make a LS regression to get gama (hedge ratio) and mu (offset term)

		#to reorder dataset col, order as same as self.holdings
		dataset = Portfolio.reorder_dataset_as_holdings(dataset, self.holdings)

		if type(dataset) == bool and  dataset == False:
			raise TypeError("Not All holdings have Price")

		#regression target y : 1st col in dataset, regresion input : reset of col start from 2nd col 
		y, x = dataset.iloc[:, 0 : 1].to_numpy(), dataset.iloc[:, 1:].to_numpy()

		# gama :class 'numpy.ndarray' [float] ; mu : class 'numpy.ndarray' [float]
		gama, mu = np.linalg.lstsq(np.hstack([x, np.ones( (len(x), 1)  ) ]), y, rcond=None)[0] # slop : gama, interception : mu

		return gama, mu


	def get_residue(self, dataset, gama, mu):
		# gama : [float] ; mu : [float]
		
		#to reorder dataset col, order as same as self.holdings
		dataset = Portfolio.reorder_dataset_as_holdings(dataset, self.holdings)

		if type(dataset) == bool and  dataset == False:
			raise TypeError("Not All holdings have Price")

		#regression target y : 1st col in dataset, regresion input : reset of col start from 2nd col 
		y, x = dataset.iloc[:, 0 : 1].to_numpy(), dataset.iloc[:, 1:].to_numpy()

		residue = y - x * gama + mu
		return residue[:,0]

	@staticmethod
	def reorder_dataset_as_holdings(dataset, holdings):
		for h in holdings:
			if '%s_c'%(h) not in dataset.columns :
				return False

		target_columns_list =  dataset.columns.tolist()
		[ target_columns_list.remove('%s_c'%(h)) for h in holdings]

		target_columns_list = [ '%s_c'%(h) for h in holdings] + target_columns_list

		return dataset.reindex( columns = target_columns_list)


	@staticmethod
	def sliding_window_split(df, window_size):
		'''
		input a df indexed as date, split into smaller date frame with length of windows size 
		'''
		output = []
		for i in range(window_size, len(df)):
			output.append( df.iloc[ i - window_size : i ]  )

		return output 
			
if __name__ == '__main__':
	Portfolio(holdings = ['SQQQ','TQQQ'], price_starting_date = '2015-01-02')

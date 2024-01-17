import numpy as np
import pandas as pd
class Preprocessor(object):
	"""docstring for Preprocessor"""
	def __init__(self, arg):
		super(Preprocessor, self).__init__()
		self.arg = arg
		
	@staticmethod
	def calculate_return(price, return_column = 'c' , base_column = 'c'):
		'''
		to calculate return with close of last time priod 
		'''
		# price['base_last_period'] = price[base_column].shift(1) #with copy warning
		price = price.assign(base_last_period=price[base_column].shift(1))
		price['re'] = ( price[ return_column ].astype('float') - price['base_last_period'].astype('float') ) / price['base_last_period'].astype('float')
		return price['re'].values


	@staticmethod
	def moving_average(df, column_name, window_length = 5):
		df = df[column_name].values
		output = []
		for i in range(0, len(df)):
			if i < window_length:
				output.append(np.nan)
			else:
				output.append(np.mean(df[ i - window_length : i ]))
		return pd.DataFrame(output)


	@staticmethod
	def moving_mean_n_std(df, column_name, window_length = 30):
		df = df[column_name].values
		mean, std = [], []
		for i in range(0, len(df)):
			if i < window_length:
				mean.append(np.nan)
				std.append(np.nan)
			else:
				mean.append(np.mean(df[ i - window_length : i ]))
				std.append( np.std(df[ i - window_length : i ]) )
		return pd.DataFrame(mean), pd.DataFrame(std)


	@staticmethod
	def standardlization(df):
		return (df - df.mean()) / df.std(), df.mean(), df.std()

	@staticmethod
	def reverse_stanadrdlization(df, mean, std):
		return (df * std) - mean


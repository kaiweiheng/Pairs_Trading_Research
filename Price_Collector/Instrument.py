import sys
import logging
import datetime

sys.path.append("Price_Collector")
from obb_Price_Collector import *
from Preprocessor import *

class Instrument(object):
	"""docstring for Instrument"""
	def __init__(self, **arg):
		super(Instrument, self).__init__()
		self.arg = arg

		self.instrument_code = arg['instrument_code']
		self.price_starting_date = arg['price_starting_date']
		self.data_source = 'obb'


		self.raw_price = obb_Price_Collector.get_price_for_a_stock(self.instrument_code, start_date = self.price_starting_date)	
		self.price = self.add_inst_into_col(self.raw_price[['date','c']])

		#calculate standardlised price
		self.stad_price = self.price.copy()
		self.stad_price['c'], self.price_mean, self.price_std = Preprocessor.standardlization(self.raw_price['c'])
		self.stad_price = self.add_inst_into_col(self.stad_price[['date','c']])

		#calculate return
		self.price_return = self.raw_price
		self.price_return['c_re'] = Preprocessor.calculate_return(self.raw_price)
		self.price_return = self.price_return[['date','c_re']].dropna()
		self.price_return = self.add_inst_into_col(self.price_return)


		#calculate standardlised return
		self.stad_price_return = self.price_return.copy()
		self.stad_price_return['c_re'], self.return_mean, self.return_std = Preprocessor.standardlization(self.price_return['%s_c_re'%(self.instrument_code)])
		self.stad_price_return = self.add_inst_into_col(self.stad_price_return[['date','c_re']])


	def add_inst_into_col(self, df):
		# df = df.copy()
		for col in df.columns:
			if col == 'date':
				continue
			df = df.rename(columns={col : "%s_%s"%(self.instrument_code,col) })		
		return df

if __name__ == '__main__':
	Instrument(instrument_code = 'SQQQ', price_starting_date = '2015-01-01')
	Instrument(instrument_code = 'TQQQ', price_starting_date = '2015-01-01')

	# Instrument(instrument_code = 'VIXY', price_starting_date = '2015-01-01')
	# Instrument(instrument_code = 'VXX', price_starting_date = '2015-01-01')	
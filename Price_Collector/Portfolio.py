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
			self.inst_price = pd.merge(self.inst_price, inst.price, how = 'inner', left_on = 'date', right_on = 'date')


		self.inst_stad_price = self.inst_list[0].stad_price
		for inst in self.inst_list[1:]:
			self.inst_stad_price = pd.merge(self.inst_stad_price, inst.stad_price, how = 'inner', left_on = 'date', right_on = 'date')			

		self.inst_re = self.inst_list[0].price_return
		for inst in self.inst_list[1:]:
			self.inst_re = pd.merge(self.inst_re, inst.price_return, how = 'inner', left_on = 'date', right_on = 'date')

		self.inst_stad_re = self.inst_list[0].stad_price_return
		for inst in self.inst_list[1:]:
			self.inst_stad_re = pd.merge(self.inst_stad_re, inst.stad_price_return, how = 'inner', left_on = 'date', right_on = 'date')
			
if __name__ == '__main__':
	Portfolio(holdings = ['SQQQ','TQQQ'], price_starting_date = '2015-01-02')

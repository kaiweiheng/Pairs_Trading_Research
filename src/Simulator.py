import pandas as pd
import numpy as np
from Pairs_Trading_Portfolio import Pairs_Trading_Portfolio 
from Pairs_Trading_RLS_Portfolio import Pairs_Trading_RLS_Portfolio
from Pairs_Trading_Kalman_Portfolio import Pairs_Trading_Kalman_Portfolio

class Simulator(object):
	"""docstring for Simulator"""
	def __init__(self, **arg):
		super(Simulator, self).__init__()
		# self.arg = arg
		self.holdings = arg['holdings']
		self.dataset = arg['dataset']
		self.cash, self.init_cash = arg['init_amount'], arg['init_amount']

		self.simulate()

	def simulate(self):

		position_records = []
		cash_records, securities_value_records = [], []
		existed_position_flag, residue_record = 0, 0

		#deal existed position
		for index, row in self.dataset[ self.dataset['dataset_tag'] == 'test'].iterrows():
			print( 1000 * (self.cash/self.init_cash) )
			#for each row (a trading day)
			for position in position_records:
				if position.unwined_tag == 0:
					self.cash += position.update(row)

			#deal signal

			#if signal is not 0
			if row['Signal'] == 1:
				print("Engage Long")
				p = Position( holdings = self.holdings, input_row = row, cash_balance = self.cash)

				position_records.append(p)
				self.cash = self.cash - p.security_value
				

			elif row['Signal'] == -1:
				print("Engage Short")
				p = Position( holdings = self.holdings, input_row = row, cash_balance = self.cash)
				
				position_records.append(p)
				self.cash = self.cash - p.security_value
				
		return 0

class Position(object):
	"""docstring for Position"""
	def __init__(self, **arg):
		# super(Position, self).__init__()
		self.holdings = arg['holdings']
		self.input_row  = arg['input_row']
		self.unwined_tag = 0
		self.amount = arg['cash_balance'] // sum( [  self.input_row["%s_c"%(hold)] * self.input_row["%s_gama"%(hold)] for hold in self.holdings ] )

		self.security_value = sum(  [int(self.amount * self.input_row["%s_gama"%(hold)] ) * self.input_row["%s_c"%(hold)]  for hold in self.holdings ] )

		# self.update(arg['input_row'])

	def update(self, row):

		PnL = sum( [  ( self.input_row["%s_Signal"%(hold)] * self.input_row["%s_c"%(hold)] + -1 * self.input_row["%s_Signal"%(hold)] * row["%s_c"%(hold)]  ) * int(self.amount * self.input_row["%s_gama"%(hold)] )   for hold in self.holdings]  )
		# PnL = sum( [  ( -1 * self.input_row["%s_Signal"%(hold)] * self.input_row["%s_c"%(hold)] + self.input_row["%s_Signal"%(hold)] * row["%s_c"%(hold)]  ) * int(self.amount * self.input_row["%s_gama"%(hold)] )   for hold in self.holdings]  )
		

		if abs(self.input_row['residue'] - row['residue'] ) >= abs(self.input_row['residue']):
			self.unwined_tag = 1


			print(pd.concat([self.input_row, row] ,  axis = 1))
			print(self.amount)
			print(PnL)
			print("  ")
			return self.security_value + PnL

		return 0		



if __name__ == '__main__':
	# klm = Pairs_Trading_Kalman_Portfolio(holdings = ["TQQQ","SQQQ"], price_starting_date = '2015-01-01')
	# upper_bound, lower_bound = Simulator.get_std_bound(klm.dataset[ klm.dataset['is_train'] == 1 ]['residues'].values)
	# klm = Simulator(dataset = klm.dataset, holdings = ["TQQQ","SQQQ"] ,init_amount = 10000, upper_bound = upper_bound, lower_bound = lower_bound)
	# klm.simulate()


	# klm = Pairs_Trading_Kalman_Portfolio(holdings = ['EWH','EWZ'], price_starting_date = '2015-01-01')
	# upper_bound, lower_bound = Simulator.get_std_bound(klm.dataset[ klm.dataset['is_train'] == 1 ]['residues'].values)
	# klm = Simulator(dataset = klm.dataset, holdings = ['EWH','EWZ'] ,init_amount = 10000, upper_bound = upper_bound, lower_bound = lower_bound)
	# klm.simulate()

	# klm = Pairs_Trading_Kalman_Portfolio(holdings = ["VIXY","VXX"], price_starting_date = '2015-01-01')
	# upper_bound, lower_bound = Simulator.get_std_bound(klm.dataset[ klm.dataset['is_train'] == 1 ]['residues'].values)
	# klm = Simulator(dataset = klm.dataset, holdings = ["VIXY","VXX"] ,init_amount = 10000, upper_bound = upper_bound, lower_bound = lower_bound)
	# klm.simulate()

	# klm = Pairs_Trading_Kalman_Portfolio(holdings = ["QQQ","QQQM"], price_starting_date = '2015-01-03')	
	# Simulator(dataset = klm.dataset, init_amount = 10000, holdings = klm.holdings)

	# klm = Pairs_Trading_Kalman_Portfolio(holdings = ["VOO","SPY"], price_starting_date = '2015-01-03')	
	# Simulator(dataset = klm.dataset, init_amount = 100000, holdings = klm.holdings)

	klm = Pairs_Trading_Kalman_Portfolio(holdings = ["0005.HK","0011.HK"], price_starting_date = '2015-01-03')	
	Simulator(dataset = klm.dataset, init_amount = 100000, holdings = klm.holdings)

	'''
	upper_bound, lower_bound = Simulator.get_std_bound(klm.dataset[ klm.dataset['is_train'] == 1 ]['residues'].values)
	klm = Simulator(dataset = klm.dataset, holdings = ["0005.HK","0011.HK"] ,init_amount = 100000, upper_bound = upper_bound, lower_bound = lower_bound)
	klm.simulate()
	# print(klm.dataset)
	'''
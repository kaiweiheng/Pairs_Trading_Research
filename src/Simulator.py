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
		self.dataset = arg['dataset']
		self.init_amount = arg['init_amount']
		self.holdings = arg['holdings']
		self.upper_bound, self.lower_bound= arg['upper_bound'], arg['lower_bound']

	def simulate(self):
		train = self.dataset[ self.dataset['is_train'] == 1]
		dataset = self.dataset[ self.dataset['is_train'] == 0]

		position_records = []
		cash_records, securities_value_records = [], []
		existed_position_flag, residue_record = 0, 0


		for index, row in dataset.iterrows():
			position_size = self.determine_max_amount(self.init_amount ,self.holdings, row)

			if row['residues'] >= self.upper_bound and existed_position_flag == 0 :
			#TBD : define the max amount to hold
				
				p = Position( holdings = self.holdings, type = 's', open_price = row, upper_bound = self.upper_bound, lower_bound = self.lower_bound)
				existed_position_flag, residue_record = -1, row['residues']

			elif row['residues'] <= self.lower_bound and existed_position_flag == 0:
			# if (lower than lower_bound) and (no existed position)
								
				p = Position( holdings = self.holdings, type = 'l', open_price = row, upper_bound = self.upper_bound, lower_bound = self.lower_bound)
				#open new long position, long 1 holdings[0], short gama holdings[1:]
				existed_position_flag, residue_record = 1, row['residues']

			#check if unwind
			elif existed_position_flag != 0:
				is_unwing , _ = p.update(row)
				if is_unwing == 1:
					position_records += [p]
					existed_position_flag, residue_record = 0, 0

		output_keys = ['open_time','type','open_residue','close_time','update_times','close_residue','pnl']
		output_df = pd.concat( [ p.to_df(output_keys) for p in position_records])

		output_df.to_csv('./data/output_Kalman/%s_%s.csv'%(self.holdings[0],self.holdings[1]), index = False)
		
		# r = 1
		# for i in output_df['PnL'].values:
		# 	r =  r * (1 + i/10000)
		# print(  255 * (r - 1) / len(dataset) )
		return 0

	@staticmethod
	def determine_max_amount(balance, holdings, prices):
		gamas = [1] + [ g for g in prices['gama']]
		gamas = { holdings[i] : gamas[i]  for i in range(0, len(holdings))}
		prices = { h : prices['%s_c'%(h)] for h in holdings}

		value_for_one_portion =  sum([ abs(gamas[h] * prices[h]) for h in holdings ])
		max_int_portion = balance // value_for_one_portion
		gamas = { h : int( max_int_portion *  gamas[h] ) for h in holdings }

		return gamas

	@staticmethod
	def determine_amount(balance, holdings, prices):
		gamas = [1] + [ g for g in prices['gama']]
		gamas = { holdings[i] : gamas[i]  for i in range(0, len(holdings))}

		return gamas


	@staticmethod
	def get_std_bound(train_residues, factor = 2.0):
		return np.mean(train_residues) + factor * np.std(train_residues), np.mean(train_residues) - factor * np.std(train_residues)

	@staticmethod
	def get_unwing_bound(train_residues, factor = 1.0):
		return np.mean(train_residues) + factor * np.std(train_residues), np.mean(train_residues) - factor * np.std(train_residues)

class Position(object):
	"""docstring for Position"""
	def __init__(self, **arg):
		super(Position, self).__init__()
		# self.arg = arg
		self.holdings = arg['holdings']
		self.type = arg['type']

		self.open_time, self.close_time = arg['open_price'].name, 0 
		self.open_price = { i : arg['open_price']['%s_c'%(i)] for i in self.holdings}
		
		self.open_residue = arg['open_price']['residues']
		self.close_residue = 0
		if self.type == 'l':
		#open new long position, long 1 holdings[0], short gama holdings[1:]
			self.gama  = [1] + [ -1 * g for g in arg['open_price']['gama'] ]
		elif self.type == 's':
		#open new short position, short 1 holdings[0], long gama holdings[1:]
			self.gama  = [-1] + [  g for g in arg['open_price']['gama'] ]
		self.gama = { self.holdings[i] : self.gama[i] for i in range(0,len(self.holdings))}

		self.upper_bound, self.lower_bound = arg['upper_bound'], arg['lower_bound']
		
		self.update_times = -1
		self.pnl = 0

		# _, self.open_value = self.update(arg['open_price'])
		self.open_value = sum([ abs(self.open_price[i] * self.gama[i])  for i in self.holdings ]) + self.get_pnl(arg['open_price'])
		self.update_times += 1

	def update(self, current_price):
		self.current_value = sum([ abs(self.open_price[i] * self.gama[i])  for i in self.holdings ]) + self.get_pnl(current_price)
		self.update_times += 1

		print("%s open at %s value %.2f open r %.3f ; now at %s value %.2f current r %.3f ; profit %.3f %.1f bps \n"%
			(self.type, self.open_time,  self.open_value, self.open_residue,
			current_price.name, self.current_value, current_price['residues'],
			self.current_value - self.open_value, 10000 * (self.current_value - self.open_value) / self.open_value  ))	


		if self.type == 'l' and current_price['residues'] - self.open_residue >= abs(self.lower_bound):

			self.close_time, self.close_residue = current_price.name, current_price['residues']
			self.pnl = 10000 * (self.current_value - self.open_value) / self.open_value

			return 1 , self.current_value

		elif self.type == 's' and self.open_residue - current_price['residues']  >= abs(self.upper_bound):
			
			self.close_time, self.close_residue = current_price.name, current_price['residues']
			self.pnl = 10000 * (self.current_value - self.open_value) / self.open_value
			
			return 1 , self.current_value


		return 0 , self.current_value

	def get_cost(self, current_price):
		return 0

	def get_pnl(self, current_price):
		self.current_price = { i : current_price['%s_c'%(i)] for i in self.holdings}
		return sum([ self.gama[h] * ( self.current_price[h] - self.open_price[h] ) for h in self.holdings])

	def to_df(self, output_keys):
		# output_dict = dict()
		# for key in output_keys:
		# 	output_dict[key] = [getattr(self, key)]
		# return pd.DataFrame.from_dict(output_dict) 
		return pd.DataFrame.from_dict( { key : [getattr(self, key)] for key in output_keys} )


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

	klm = Pairs_Trading_Kalman_Portfolio(holdings = ["0005.HK","0011.HK"], price_starting_date = '2015-01-03')	
	print(klm.dataset)

	'''
	upper_bound, lower_bound = Simulator.get_std_bound(klm.dataset[ klm.dataset['is_train'] == 1 ]['residues'].values)
	klm = Simulator(dataset = klm.dataset, holdings = ["0005.HK","0011.HK"] ,init_amount = 100000, upper_bound = upper_bound, lower_bound = lower_bound)
	klm.simulate()
	# print(klm.dataset)
	'''
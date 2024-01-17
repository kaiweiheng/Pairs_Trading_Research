import sys
import logging
import datetime
import numpy as np
import pandas as pd

sys.path.append("Price_Collector")
from Portfolio import Portfolio

sys.path.append("Analysis")
from Simple_Analysis import Simple_Analysis


from statsmodels.tsa.stattools import coint 
from scipy.stats import pearsonr, spearmanr

from statsmodels.tsa.stattools import adfuller

class Pairs_Trading_Portfolio(Portfolio):
	"""docstring for Pairs_Trading_Portfolio"""
	def __init__(self, **arg):
		super().__init__( **arg )

		#cointegrate test price returns 
		# _, pv_coint, _ = coint( self.inst_re['%s_c_re'%(self.holdings[0])], self.inst_re['%s_c_re'%(self.holdings[1])] )
		# corr, pv_corr = pearsonr(self.inst_re['%s_c_re'%(self.holdings[0])], self.inst_re['%s_c_re'%(self.holdings[1])])

		# #cointegrate test log price
		# _, pv_coint, _ = coint( self.inst_price['%s_c'%(self.holdings[0])], self.inst_price['%s_c'%(self.holdings[1])] )
		# corr, pv_corr = pearsonr(self.inst_price['%s_c'%(self.holdings[0])], self.inst_price['%s_c'%(self.holdings[1])])


		#cointegrate test log price
		# _, pv_coint, _ = coint( self.inst_stad_price['%s_c'%(self.holdings[0])], self.inst_stad_price['%s_c'%(self.holdings[1])] )
		# corr, pv_corr = pearsonr(self.inst_stad_price['%s_c'%(self.holdings[0])], self.inst_stad_price['%s_c'%(self.holdings[1])])


		# print("\n %s %s Cointegration pvalue : %0.4f"%(self.holdings[0],self.holdings[1],pv_coint))
		# print("Correlation coefficient is %0.4f and pvalue is %0.4f \n"%(corr, pv_corr))		



		# x, y = self.inst_stad_price['%s_c'%(self.holdings[0]) ].to_numpy(), self.inst_stad_price[ '%s_c'%(self.holdings[1])  ].to_numpy()
		x, y = self.inst_price['%s_c'%(self.holdings[0]) ].to_numpy(), self.inst_price[ '%s_c'%(self.holdings[1])  ].to_numpy()
		A = np.vstack([x, np.ones(len(x)) ]).T
		gama, mu = np.linalg.lstsq(A, y, rcond=None)[0] # slop : gama, interception : mu

		residue = y - x * gama + mu

		print(pd.DataFrame(residue).describe())

		Simple_Analysis.plot_single_factor_graph(self.inst_stad_price['date'][300:700], residue[300:700], '%s_%s'%(self.holdings[0],self.holdings[1]) ,True)

		dftest = adfuller(residue)
		print(self.holdings)
		print(dftest)
		# dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value(%)','#Lags Used','Number of Observations Used'])		
		# for key,value in dftest[4].items():

		# 	dfoutput['Critical Value (%s)' % key] = value
		# 	print(dfoutput)


		print("   ")



if __name__ == '__main__':
	Pairs_Trading_Portfolio(holdings = ['QQQ','QQQM'], price_starting_date = '2015-01-01')
	Pairs_Trading_Portfolio(holdings = ['SQQQ','TQQQ'], price_starting_date = '2015-01-01')

	Pairs_Trading_Portfolio(holdings = ["VIXY","VXX"], price_starting_date = '2015-01-01')

	# Pairs_Trading_Portfolio(holdings = ["0005.HK","0011.HK"], price_starting_date = '2015-01-01')		
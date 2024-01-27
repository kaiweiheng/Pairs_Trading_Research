import os
import sys
import numpy as np
from datetime import date
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

sys.path.append("util")
from path import *

class Simple_Analysis(object):
	"""docstring for Simple_Analysis"""
	def __init__(self, arg):
		super(Simple_Analysis, self).__init__()
		self.arg = arg
	
	@staticmethod
	def return_histogram(df,  file_name, if_save = True):
		#pass in a dataframe, plot distribution of df with the colunm name as label
		#a description of each col's p value will be appended

		plt.style.use('seaborn-deep')
		n_bins = 100
		plt.rcParams['font.size'] = 7		

		text = '%s Residue Histogram\n'%(file_name)
		fig, ax = plt.subplots()
		for (column_name, column_data) in df.items():
			ax.hist( column_data.dropna().values , n_bins, histtype='step', stacked=True, fill=False, density = True, label = column_name)		

			dftest = adfuller(column_data.dropna().values)
			p_value =  dftest[1]

			text += "%s DickyFuller P-value : %.3f%%, mean : %.3f, std : %.3f \n"%(column_name, p_value*100, np.mean(column_data.dropna().values), np.std(column_data.dropna().values))

		ax.legend(loc = 'upper right',prop={'size': 8})
		ax.set_title(text)
		plt.close(fig) #not showing in the jupyter lab

		if if_save:
			output_path = os.path.join('data','output',"%s_histogram.png"%(file_name))
			check_parents_dir_exist(output_path)
			fig.savefig(output_path, dpi = 250)
		
		return ax, fig


	@staticmethod
	def plot_time_series_df(df, linestyles, colors, output_path, if_save = True):

		#pass df sorted date as index, plot and save into a graph
		date = df.index.to_list()

		plt.rcParams['font.size'] = 7		
		fig, ax = plt.subplots()
		fig.set_figheight(8)
		fig.set_figwidth(int(len(date))/10)

		acc = 0
		for (column_name, column_data) in df.items():
			y = column_data.values
			ax.plot(date, y, color = colors[acc] , linestyle = linestyles[acc] ,linewidth=0.25, label = column_name)
			acc += 1

		ax.set(xlabel='time', ylabel= df.columns[0], title='Record')
		ax.grid(color = 'w', linewidth = 0.2)
		plt.legend(loc = 'upper right',prop={'size': 8})

		ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
		# ax.xaxis.set_major_locator(mdates.DayLocator(interval=int(  len(date)/10 )))
		ax.xaxis.set_major_locator(mdates.DayLocator(interval=int(2)))
		ax.tick_params(axis='x', labelrotation=45, labelsize=5)

		plt.close(fig) #not showing in the jupyter lab
		if if_save:
			# output_path = os.path.join(output_path)
			check_parents_dir_exist(output_path)
			fig.savefig(output_path, dpi = 250)
		return ax, fig

	@staticmethod
	def plot_adf(df, output_path, if_save = True):
		maxlags = 200
		plt.style.use('seaborn-deep')
		fig, ax = plt.subplots()

		for (column_name, column_data) in df.items():
			ax.acorr(column_data.dropna().values, usevlines = False, normed = True, maxlags = maxlags, label = column_name, marker = '.', markersize = 1, lw = 2)

		ax.grid(color = 'w', linewidth = 0.2)
		ax.set(xlabel='Lags', ylabel= 'Self-Correlation')
		plt.xlim([0, maxlags])
		plt.legend(loc = 'upper right',prop={'size': 8})
		if if_save:
			# output_path = os.path.join(output_path)
			check_parents_dir_exist(output_path)
			fig.savefig(output_path, dpi = 250)
		return ax, fig

import os

# @staticmethod
def check_parents_dir_exist(path):
	parents_dir = os.path.dirname(path)
	if not os.path.exists(parents_dir):
		os.makedirs(parents_dir)		

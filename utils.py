import pandas as pd
import numpy as np

def load_dataset(file, columns, threshold):
	df = pd.read_csv(file, skiprows=3, header=None, names=columns, dtype=np.float, sep='\s+', decimal=',', index_col=False)
	for a in columns:
		df[a+'_bin'] = 0
		df.loc[df[a] > threshold, a+'_bin'] = 1
	return df
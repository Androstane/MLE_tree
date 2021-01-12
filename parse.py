import pandas as pd
from scipy.spatial.distance import pdist, squareform
import numpy as np

def parse(path):
    cnp = pd.read_csv(path + '/gt.cnp', sep = '\t')
    cnp.drop(cnp.columns[[0,1,2]], axis = 1, inplace = True) 
    cnp.columns = cnp.columns.str.replace(' ', '')
    M = cnp.values
    M = M.transpose()
    M = pdist(M, metric='hamming')
    M = squareform(M)
    leaf_name = list(cnp.columns.values) 
    d = cnp.to_dict('list')
    return path, M, leaf_name, d
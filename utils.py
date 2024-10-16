import sklearn.preprocessing
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sklearn

datapath = r'C:\Users\User\Desktop\github\Loan_Approval_Pred\data\train.csv'
test_datapath = r'C:\Users\User\Desktop\github\Loan_Approval_Pred\data\test.csv'
epochs = 200

std_scale = sklearn.preprocessing.StandardScaler()

def str_key(data_f, dict) :
    r_lst = np.array([list(map(lambda x : dict[x], data_f))])
    return r_lst
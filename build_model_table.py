import pandas as pd
import os
import re

DATA_PATH='Data/'

dis_table = pd.read_csv(DATA_PATH+"TargetDiseaseCodes.txt",sep='\t',index_col="CODE")

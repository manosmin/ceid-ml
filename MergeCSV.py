import pandas as pd
import glob
import os
import datetime
from datetime import timedelta

path1 = './sources'
all_files1 = glob.glob(os.path.join(path1 , "*.csv"))

path2 = './demand'
all_files2 = glob.glob(os.path.join(path2 , "*.csv"))

l1=[]
l2=[]

# read csv files from sources folder
for filename1 in all_files1:
    print(filename1)
    csv1 = pd.read_csv(filename1, index_col=None, header=0)
    l1.append(csv1)

# read csv files from demand folder
for filename2 in all_files2:
    print(filename2)
    csv2 = pd.read_csv(filename2, index_col=None, header=0)
    l2.append(csv2)

d = datetime.date(2019, 1, 1)
totalFiles = 1096

for i in range(0, totalFiles):
    # merge data frames
    df = pd.merge(l1[i], l2[i])
    # drop rows with duplicate values in time column
    df = df.drop_duplicates(subset='Time', keep='first')
    # export data frame as csv to merged folder
    df.to_csv( "merged/merged"+str(d)+".csv", index=False)
    d = d + timedelta(days = 1) 



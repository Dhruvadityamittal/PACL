import glob
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
# from utils import resize

DATAFILES = "MHEALTHDATASET/"
OUTDIR = "data/"

if not os.path.isdir(OUTDIR):
    os.makedirs(OUTDIR)

data_files = os.listdir(DATAFILES)
X = []
Y = []
P = []

for data_file in tqdm(data_files):
    if(data_file[-3:]=='log'):
        subject_id = data_file.split(".")[0].split("t")[2]
        with open(DATAFILES+data_file, 'r') as subject_data:
            for data in subject_data:
                data_splitted = data.split("\t")
                if(data_splitted[-1].strip() != '0'):
                    X.append([data_splitted[14],data_splitted[15],data_splitted[16]])
                    Y.append(data_splitted[-1].strip())
                    P.append(subject_id)
               
                
   
X = np.array(X)
Y = np.array(Y)
P = np.array(P)



os.system(f"mkdir -p {OUTDIR}")
np.save(os.path.join(OUTDIR, "X_unclean"), X)
np.save(os.path.join(OUTDIR, "Y_unclean"), Y)
# np.save(os.path.join(OUTDIR, "time"), T)
np.save(os.path.join(OUTDIR, "pid_unclean"), P)

print(X.shape)
print("X shape:", X.shape)
print("Y distribution:", len(set(Y)))
print(pd.Series(Y).value_counts())
print("User distribution:", len(set(P)))
print(pd.Series(P).value_counts())



# with open(file_path, 'r') as file:

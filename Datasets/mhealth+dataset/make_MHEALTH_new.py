import glob
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
# from utils import resize
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--standardize', default=True, type=bool, help='Standarize?') 

args = parser.parse_args()

DATAFILES = "MHEALTHDATASET/"
OUTDIR = "data/"
WINDOW_SEC = 2
OVERLAP = 1.5
FREQ = 50

WINDOW_OVERAL = OVERLAP*FREQ
WINDOW_LEN = WINDOW_SEC*FREQ

WINDOW_STEP = int(WINDOW_LEN - WINDOW_OVERAL)

if not os.path.isdir(OUTDIR):
    os.makedirs(OUTDIR)

data_files = os.listdir(DATAFILES)
X = []
Y = []
P = []
standardizing = True

def is_good_quality(w):
    ''' Window quality check '''

    if w.isna().any().any():
        return False

    if len(w) != WINDOW_LEN:
        return False

    if len(w['23'].unique()) > 1:
        
        return False
    else:
        if(w['23'].unique()[0]==0):
            return False
    # w_start, w_end = w.index[0], w.index[-1]
    # w_duration = w_end - w_start
    # target_duration = pd.Timedelta(10, 's')
    # if np.abs(w_duration - target_duration) > 0.01 * target_duration:
    #     return False

    return True

for data_file in tqdm(data_files):
    if(data_file[-3:]=='log'):
        subject_id = data_file.split(".")[0].split("t")[2]
        data = pd.read_csv(DATAFILES+data_file, delimiter='\t', header=None)
        data.columns = [str(i) for i in range(data.shape[1])]
        # print(data.columns)
        
        for i in range(0, len(data), WINDOW_STEP):
            
            w = data.iloc[i:i + WINDOW_LEN]
            if not is_good_quality(w):
                # print("Bad Quality")
                continue
            x = w[['0','1','2', '5', '6', '7','14', '15', '16']].values
            y = np.unique(w['23'].values)[0]
            

            X.append(x)
            Y.append(y)
            
            P.append(subject_id)
  
        # with open(DATAFILES+data_file, 'r') as subject_data:
        #     for data in subject_data:
        #         data_splitted = data.split("\t")
        #         if(data_splitted[-1].strip() != '0'):
        #             X.append([data_splitted[14],data_splitted[15],data_splitted[16]])
        #             Y.append(data_splitted[-1].strip())
        #             P.append(subject_id)
               
                
   
# X = np.array(X)

# # Standardizing X
if(args.standardize):
    X = np.array(X, dtype=np.float32)
    m = np.mean(X, axis=0)
    X -= m
    std = np.std(X, axis=0)
    std += 0.000001
    X /= (std * 2)  # 2 is for having smaller values
    name_x = "X_Standardized"
    name_y = "Y_Standardized"
    name_p = "pid_Standardized"
else:
    name_x = "X"
    name_y = "Y"
    name_p = "pid"

Y = np.array(Y)
P = np.array(P)
os.system(f"mkdir -p {OUTDIR}")
np.save(os.path.join(OUTDIR, name_x), X)
np.save(os.path.join(OUTDIR, name_y), Y)
# np.save(os.path.join(OUTDIR, "time"), T)
np.save(os.path.join(OUTDIR, name_p), P)

print(X.shape)
print("X shape:", X.shape)
print("Y shape:", Y.shape)

print("Y distribution:", len(set(Y)))
print(pd.Series(Y).value_counts())
print("User distribution:", len(set(P)))
print(pd.Series(P).value_counts())



# with open(file_path, 'r') as file:

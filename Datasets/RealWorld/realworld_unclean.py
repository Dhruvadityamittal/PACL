"""
first run realworld_raw_preprocess.py
"""
import glob
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from utils import resize

DEVICE_HZ = 50  # Hz
WINDOW_SEC = 2  # seconds
WINDOW_OVERLAP_SEC = 0.5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 100  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)
DATAFILES = "imu/*.npy"
OUTDIR = "realworld_30hz_w10/"
LABEL_NAMES = [
    "jumping",
    "climbingup",
    "climbingdown",
    "lying",
    "running",
    "sitting",
    "standing",
    "walking",
]
BODY_PARTS = ["chest", "forearm", "head", "shin", "thigh", "upperarm", "waist"] 
# start_ind = BODY_PARTS.index("forearm") * 3 # Forearm data starts after chest data
start_ind = (BODY_PARTS.index("forearm") * 4) +1# Forearm data starts after chest data
# For one 


def is_numpy_array_good_quality(window):
    """Window quality check"""

    if np.isnan(window).any():
        return False

    if len(window) != WINDOW_LEN:
        return False

    return True


X, Y, S, P, T = [], [], [], [], []
for datafile in tqdm(glob.glob(DATAFILES)):

    pid, sess_id, class_name, _ = datafile.split("\\")[-1].split(".")


    
    data = np.load(datafile)[
        :, start_ind : start_ind + 3
    ]  # data corresponding to forearm
    data_time = np.load(datafile)[
        :, start_ind-1
    ]
    
    if(len(X)==0):
        X.append(data)
        T.append(data_time)
        X = X[0]
        T = T[0]
    else:
        X = np.concatenate([X,data])
        T = np.concatenate([T,data_time])

    Y = np.concatenate([Y,len(data)*[class_name]])
    S = np.concatenate([S,len(data)*[sess_id]])
    P = np.concatenate([P,len(data)*[pid]])

    # # Y.append(len(data)*[class_name])
    # S.append(len(data)*[sess_id])
    # P.append(len(data)*[pid])

    
    
    

#     for i in range(0, len(data), WINDOW_STEP_LEN):
#         window = data[i : i + WINDOW_LEN]
#         if not is_numpy_array_good_quality(window):
#             continue
#         X.append(window)
#         Y.append(class_name)
#         S.append(sess_id)
#         P.append(pid)
# exit()

X = np.asarray(X)

Y = np.asarray(Y).flatten()
S = np.asarray(S).flatten()
P = np.asarray(P).flatten()
T = np.asarray(T).flatten()


# fixing unit to g
# X = X / 9.81
# downsample to 30 Hz
# X = resize(X, TARGET_WINDOW_LEN)/

os.system(f"mkdir -p {OUTDIR}")
np.save(os.path.join(OUTDIR, "X_unclean"), X)
np.save(os.path.join(OUTDIR, "Y_unclean"), Y)
np.save(os.path.join(OUTDIR, "session_id_unclean"), S)
np.save(os.path.join(OUTDIR, "pid_unclean"), P)
np.save(os.path.join(OUTDIR, "time_unclean"), T)

print(f"Saved in {OUTDIR}")
print("X shape:", X.shape)
print("Y distribution:", len(set(Y)))
print(pd.Series(Y).value_counts())
print("User distribution:", len(set(P)))
print(pd.Series(P).value_counts())
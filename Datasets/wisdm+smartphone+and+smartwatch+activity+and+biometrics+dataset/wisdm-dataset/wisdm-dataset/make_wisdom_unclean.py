import glob
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from utils import resize

print("Hello")

DEVICE_HZ = 20  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%  # Window Tolerence
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)
DATAFILES = "/Users/catong/repos/video-imu/data/"
DATAFILES = DATAFILES + "wisdm/wisdm-dataset/raw/watch/accel/*.txt"
DATAFILES = "raw/watch/accel/*.txt"
OUTDIR = "wisdm_30hz_w10/"


label_dict = {}
label_dict["walking"] = "A"
label_dict["jogging"] = "B"
label_dict["stairs"] = "C"
label_dict["sitting"] = "D"
label_dict["standing"] = "E"
label_dict["typing"] = "F"
label_dict["teeth"] = "G"
label_dict["soup"] = "H"
label_dict["chips"] = "I"
label_dict["pasta"] = "J"
label_dict["drinking"] = "K"
label_dict["sandwich"] = "L"
label_dict["kicking"] = "M"
label_dict["catch"] = "O"
label_dict["dribbling"] = "P"
label_dict["writing"] = "Q"
label_dict["clapping"] = "R"
label_dict["folding"] = "S"
code2name = {code: name for name, code in label_dict.items()}  # Just interchanging keys to values




def is_good_quality(w):
    """Window quality check"""

    if w.isna().any().any():
        return False

    if len(w) != WINDOW_LEN:
        return False

    # if len(w['annotation'].unique()) > 1:
    # return False


    w_start, w_end = w.index[0], w.index[-1]
    w_duration = w_end - w_start
    target_duration = pd.Timedelta(WINDOW_SEC, "s")
    if np.abs(w_duration - target_duration) > WINDOW_TOL * target_duration:
        return False

    return True


# annolabel = pd.read_csv(ANNOLABELFILE, index_col='annotation')

X, Y, T, P, = (
    [],
    [],
    [],
    [],
)


def tmp(my_x):
    return float(my_x.strip(";"))


column_names = ["pid", "code", "time", "x", "y", "z"]
l = 0
l1 = 0
for datafile in tqdm(glob.glob(DATAFILES)):
    columns = ["pid", "class_code", "time", "x", "y", "z"]
    one_person_data = pd.read_csv(
        datafile,
        sep=",",
        header=None,
        converters={5: tmp},
        names=column_names,
        parse_dates=["time"],
        index_col="time",
    )
    one_person_data.index = pd.to_datetime(one_person_data.index)
    period = int(round((1 / DEVICE_HZ) * 1000_000_000))
    # one_person_data.resample(f'{period}N', origin='start').nearest(limit=1)
    code_to_df = dict(tuple(one_person_data.groupby("code")))  # Returns all the collums for a specific class in a dict
    pid = int(one_person_data["pid"][0])


    for code, data in code_to_df.items():

        try:
            data = data.resample(f"{period}N", origin="start").nearest(limit=1)  #Each sample is collected in 50ms -> 5e-7 ns
        except ValueError:
            if pid == 1629:
                data = data.drop_duplicates()
                data = data.resample(f"{period}N", origin="start").nearest(
                    limit=1
                )
                pass
        
        # Just to know the number of columns skipped
        # l = l+len(data)
        # data = data.dropna(subset=['code'])
        # l1 = l1+len(data)
    
        # print(code,data.index[0], data.index[-1])

        if(len(X)==0):
            X = data[['x','y','z']].values
            Y = [code2name[code]]*len(data)
            P = [pid]*len(data)
            T = data.index
         
        else:
            X = np.concatenate([X,data[['x','y','z']].values])
            Y = np.concatenate([Y,[code2name[code]]*len(data)])
            P = np.concatenate([P,[pid]*len(data)])
            T = np.concatenate([T,data.index])
        
           
        # for i in range(0, len(data), WINDOW_STEP_LEN):
        #     w = data.iloc[i : i + WINDOW_LEN]

        #     if not is_good_quality(w):
        #         continue

        #     x = w[["x", "y", "z"]].values
        #     t = w.index[0].to_datetime64()

        #     X.append(x)
        #     Y.append(code2name[code])
        #     T.append(t)
        #     P.append(pid)

X = np.asarray(X)
Y = np.asarray(Y)
T = np.asarray(T)
P = np.asarray(P)
print(Y.shape)


# # fixing unit to g
# X = X / 9.81
# # downsample to 30 Hz
# X = resize(X, TARGET_WINDOW_LEN)


os.system(f"mkdir -p {OUTDIR}")
np.save(os.path.join(OUTDIR, "X_unclean"), X)
np.save(os.path.join(OUTDIR, "Y_unclean"), Y)
np.save(os.path.join(OUTDIR, "time_unclean"), T)
np.save(os.path.join(OUTDIR, "pid_unclean"), P)

print(f"Saved in {OUTDIR}")
print("X shape:", X.shape)
print("Y distribution:", len(set(Y)))
print(pd.Series(Y).value_counts())
print("User distribution:", len(set(P)))
print(pd.Series(P).value_counts())
# print(l,l1)
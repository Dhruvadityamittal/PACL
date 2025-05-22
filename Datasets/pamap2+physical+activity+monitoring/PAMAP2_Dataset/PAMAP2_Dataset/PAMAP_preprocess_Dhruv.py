import numpy as np
from scipy import stats as s
from scipy import constants
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--standardize', default=True, type=bool, help='Standarize?') 
args = parser.parse_args()

root_folder = 'Optional/'
root_folder_files = os.listdir('Optional/')
print("Loading Optional")
for root_folder_file in root_folder_files:
    datContent = [i.strip().split() for i in open(root_folder + root_folder_file).readlines()]
    datContent = np.array(datContent)
    print(np.unique(datContent[:,1]))

files_optional = [root_folder + root_folder_file  for root_folder_file in root_folder_files] 

print("Loading Protocol")
root_folder = 'Protocol/'
root_folder_files = os.listdir('Protocol/')

for root_folder_file in root_folder_files:
    datContent = [i.strip().split() for i in open(root_folder + root_folder_file).readlines()]
    datContent = np.array(datContent)
    print(np.unique(datContent[:,1]))

files_protocol= [root_folder + root_folder_file  for root_folder_file in root_folder_files] 

all_files = files_protocol + files_optional  

WINDOW_SEC = 2
WINDOW_OVERLAP_SEC = 1.5
DEVICE_HZ = 100
WINDOW_LEN = WINDOW_SEC* DEVICE_HZ
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN

X = []
Y = []
pid = []

label_idx = 1
timestamp_idx = 0  # idx 2-> Heart rate idx 3 ->IMU Hand Temperature 
x_idx_1, y_idx_1, z_idx_1 = 4, 5, 6
x_idx_2, y_idx_2, z_idx_2 = 21, 22, 23
x_idx_3, y_idx_3, z_idx_3 = 38, 39, 40


for data_file in all_files[0:]:
    datContent = [i.strip().split() for i in open(data_file).readlines()]
    datContent = np.array(datContent)
    

    datContent = datContent[:,[timestamp_idx, label_idx, x_idx_1, y_idx_1, z_idx_1, x_idx_2, y_idx_2, z_idx_2, x_idx_3, y_idx_3, z_idx_3]]
    datContent = datContent.astype(float)

    datContent = datContent[~np.isnan(datContent).any(axis=1)]
    

    person_id = data_file.split('/')[1].strip().split('.dat')[0].strip().split('subject')[1]
    

    for i in range(0, len(datContent),WINDOW_STEP_LEN ):
        if(datContent[i: i+ WINDOW_LEN, [2, 3,4, 5,6,  7, 8,9, 10 ]].shape[0] == WINDOW_LEN):
            X.append(datContent[i: i+ WINDOW_LEN, [2, 3,4, 5,6,  7, 8,9, 10 ]])  #x_idx_2, y_idx_2, z_idx_2, x_idx_3, y_idx_3, z_idx_3
            Y.append(datContent[i: i+ WINDOW_LEN, 1])
            pid.append(person_id)
    

X = np.array(X)
Y = np.array(Y)
pid = np.array(pid)

def clean_up_label(X, labels, pid):
    # 1. remove rows with >50% zeros
    
    sample_count_per_row = labels.shape[1]  # number of windows

    rows2keep = np.ones(labels.shape[0], dtype=bool)
    transition_class = 0
    for i in range(labels.shape[0]):
        row = labels[i, :]
        if np.sum(row == transition_class) > 0.5 * sample_count_per_row:
            rows2keep[i] = False

    labels = labels[rows2keep]
    X = X[rows2keep]
    pid = pid[rows2keep]

    
    # 2. majority voting for label in each epoch
    final_labels = []
    for i in range(labels.shape[0]):
        row = labels[i, :]
        final_labels.append(s.mode(row)[0])
    final_labels = np.array(final_labels, dtype=int)

    final_label_filter = (final_labels!=0)

    X = X[final_label_filter]
    
    final_labels = final_labels[final_label_filter]
    pid = pid[final_label_filter]

    # print("Clean X shape: ", X.shape)
    # print("Clean y shape: ", final_labels.shape)
    return X, final_labels, pid


current_X, current_y, current_pid = clean_up_label(X, Y, pid)

current_y = current_y.flatten()
current_X = current_X / constants.g

# For standardizing.
if(args.standardize):
    current_X = np.array(current_X, dtype=np.float32)
    m = np.mean(current_X, axis=0)
    current_X -= m
    std = np.std(current_X, axis=0)
    std += 0.000001
    current_X /= (std * 2)  # 2 is for having smaller values
    name_x = "X_Standardized"
    name_y = "Y_Standardized"
    name_p = "pid_Standardized"
else:
    name_x = "X"
    name_y = "Y"
    name_p = "pid"

np.save(f'{name_x}.npy', current_X)
np.save(f'{name_y}.npy', current_y)
np.save(f'{name_p}.npy', current_pid)

# clip_value = 3
# current_X = np.clip(current_X, -clip_value, clip_value)
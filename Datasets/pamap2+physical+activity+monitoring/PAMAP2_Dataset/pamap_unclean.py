from scipy import stats as s
from scipy import constants
import os
from tqdm import tqdm
import numpy as np
import glob

def get_data_content(data_path):
    # read flash.dat to a list of lists
    
    datContent = [i.strip().split() for i in open(data_path).readlines()]

    datContent = np.array(datContent)
    label_idx = 1
    timestamp_idx = 0  # idx 2-> Heart rate idx 3 ->IMU Hand Temperature 
    x_idx = 4
    y_idx = 5
    z_idx = 6

    
    index_to_keep = [timestamp_idx, label_idx, x_idx, y_idx, z_idx]
    # 3d +- 16 g

    datContent = datContent[:, index_to_keep]
    datContent = datContent.astype(float)

    # datContent = datContent[~np.isnan(datContent).any(axis=1)] # Removing the entire row if any column is None
    return datContent

def content2x_and_y(data_content, epoch_len=30, sample_rate=100, overlap=15):

    # sample_count = int(np.floor(len(data_content) / (epoch_len * sample_rate))) # Number of samples if we take windowlen of 30 sec and sampling rate of 30hz
    

    # Ignoring the timestamp
    sample_label_idx = 1
    sample_x_idx = 2
    sample_y_idx = 3
    sample_z_idx = 4
    sample_time_idx =0

    # sample_limit = sample_count * epoch_len * sample_rate
    # data_content = data_content[:sample_limit, :]

    label = data_content[:, sample_label_idx]
    x = data_content[:, sample_x_idx]
    y = data_content[:, sample_y_idx]
    z = data_content[:, sample_z_idx]
    time = data_content[:,sample_time_idx]
    # to make overlapping window
    # offset = overlap * sample_rate 
    # shifted_label = data_content[offset:-offset, sample_label_idx]
    # shifted_x = data_content[offset:-offset:, sample_x_idx]
    # shifted_y = data_content[offset:-offset:, sample_y_idx]
    # shifted_z = data_content[offset:-offset:, sample_z_idx]

    
    # shifted_label = shifted_label.reshape(-1, epoch_len * sample_rate)
    # shifted_x = shifted_x.reshape(-1, epoch_len * sample_rate, 1)
    # shifted_y = shifted_y.reshape(-1, epoch_len * sample_rate, 1)
    # shifted_z = shifted_z.reshape(-1, epoch_len * sample_rate, 1)
    # shifted_X = np.concatenate([shifted_x, shifted_y, shifted_z], axis=2)

    # label = label.reshape(-1, epoch_len * sample_rate)
    # x = x.reshape(-1, epoch_len * sample_rate, 1)
    # y = y.reshape(-1, epoch_len * sample_rate, 1)
    # z = z.reshape(-1, epoch_len * sample_rate, 1)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    time = time.reshape(-1,1)
    # print(x.shape,y.shape,z.shape)
    
    X = np.concatenate([x, y, z], axis=1)
    

    
    # X = np.concatenate([X, shifted_X])

    # label = np.concatenate([label, shifted_label]) # Each row contains labels for 3000 windows
    
    
    return X, label, time

def process_all(file_paths, X_path, y_path, pid_path, time_path,epoch_len, overlap):
    X = []
    y = []
    pid = []
    T = []

    for file_path in tqdm(file_paths):
        subject_id = int(file_path.split("/")[-1][-7:-4])
        datContent = get_data_content(file_path)
        current_X, current_y, time = content2x_and_y(
            datContent, epoch_len=epoch_len, overlap=overlap
        )
        ids = np.full(
            shape=len(current_y), fill_value=subject_id, dtype=np.int32
        )

        print(current_X.shape,time.shape)
        if len(X) == 0:
            X = current_X
            y = current_y
            pid = ids
            T = time
        else:
            X = np.concatenate([X, current_X])
            y = np.concatenate([y, current_y])
            pid = np.concatenate([pid, ids])
            T =  np.concatenate([T, time])

    y = y.flatten()
    # X = X / constants.g  # convert to unit of g
    # clip_value = 3
    # X = np.clip(X, -clip_value, clip_value)
    print(np.unique(pid))
    # Keep only 8 activities that everyone has
    # y_filter = (
    #     (y == 1)
    #     | (y == 2)
    #     | (y == 3)
    #     | (y == 4)
    #     | (y == 12)
    #     | (y == 13)
    #     | (y == 16)
    #     | (y == 17)
    # )
    # X = X[y_filter]
    # y = y[y_filter]
    # pid = pid[y_filter]

    print(len(X),len(y),len(pid),len(T))
    np.save(X_path, X)
    np.save(y_path, y)
    np.save(pid_path, pid)
    np.save(time_path, T)


def get_write_paths(data_root):
    X_path = os.path.join(data_root, "X_unclean.npy")
    y_path = os.path.join(data_root, "Y_unclean.npy")
    pid_path = os.path.join(data_root, "pid_unclean.npy")
    time_path = os.path.join(data_root, "time_unclean.npy")
    
    return X_path, y_path, pid_path, time_path


def main():
    data_root = "PAMAP2_Dataset/"

    data_path = data_root + "Protocol/"
    protocol_file_paths = glob.glob(data_path + "*.dat")
    data_path = data_root + "Optional/"
    optional_file_paths = glob.glob(data_path + "*.dat")
    file_paths = protocol_file_paths + optional_file_paths

    data_root = "PAMAP2_Dataset/"
    X_path, y_path, pid_path, time_path = get_write_paths(data_root)
    epoch_len = 30
    overlap = 15
    process_all(file_paths, X_path, y_path, pid_path,time_path, epoch_len, overlap)
    print("Saved X to ", X_path)
    print("Saved y to ", y_path)
    
if __name__ == "__main__":
    main()
import numpy as np


def drop_duplicates(x):
    x = np.asarray(x)
    rolled = np.roll(x, -1)
    x = x[np.where(x!=rolled)[0]]
    
    return x

def split_array(arr, step=1000):
    arr = np.asarray(arr)
    split_arrays = []
    for i in range(0, arr.shape[0] + step, step):
        split_arrays.append(arr[i:i+step])
    return split_arrays
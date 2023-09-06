import numpy as np


def drop_duplicates(x):
    """ Remove consecutive duplicate elements from a NumPy array.

    Parameters:
    -----------
    x : numpy.ndarray
        The input NumPy array from which consecutive duplicates will be removed.

    Returns:
    --------
    numpy.ndarray
        A NumPy array with consecutive duplicate elements removed.

    Notes:
    ------
    - This function removes consecutive duplicate elements from the input NumPy array `x`.
    - It compares each element with the next element and keeps only the first occurrence.
    """
    x = np.asarray(x)
    rolled = np.roll(x, -1)
    x = x[np.where(x!=rolled)[0]]
    
    return x

def split_array(arr, step=1000):
    """ Split a NumPy array into smaller sub-arrays of a specified step size.

    Parameters:
    -----------
    arr : numpy.ndarray
        The input NumPy array to be split.
    step : int, optional
        The size of each sub-array, by default 1000.

    Returns:
    --------
    list
        A list of NumPy sub-arrays obtained by splitting the input array.

    Notes:
    ------
    - This function splits a NumPy array `arr` into smaller sub-arrays of a specified size `step`.
    - It generates a list of sub-arrays, and each sub-array contains at most `step` elements.
    """
    arr = np.asarray(arr)
    split_arrays = []
    for i in range(0, arr.shape[0] + step, step):
        split_arrays.append(arr[i:i+step])
    return split_arrays
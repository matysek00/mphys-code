
import numpy as np
import sparse
from scipy.spatial.distance import pdist, squareform
from dscribe.descriptors import SOAP


def calculate_soap(data):
    species = ["H", "C"]
    r_cut = 6.0
    n_max = 8
    l_max = 8

    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        periodic=True,
        average='inner',
        sparse = True)

    soap_data = soap.create(data)
    return soap_data


def get_DM(file: str, test_size: int) -> np.array:
    """create distance matrix from soap data
    
    Parameters:
    file: .npz file containing soap data
    test_size: how many of the last trajectories should be ignored
    """
    
    soap_data = load_soap(file, test_size)    
    DM = Distance_matrix(soap_data)
    return DM


def Distance_matrix(data):
    # Use cosine distance acording to 
    # https://pubs.rsc.org/en/content/articlelanding/2016/CP/C6CP00415F
    
    DM = np.sqrt(2)*squareform(pdist(data, 'cosine'))
    return DM


def load_soap(file: str, test_size: int):
    """load soap data from a file
    
    Parameters:
    file: .npz file containing soap data
    test_size: how many of the last trajectories should be ignored   
    """

    soap_data = sparse.load_npz(file)

    if test_size == 0:
        dense = soap_data.todense()
    else:
        dense = soap_data.todense()[:-test_size]

    return dense 


def getGreedyPerm(DM: np.array):
    """
    A Naive O(N^2) algorithm to do furthest points sampling
    (Copied from:
    https://gist.github.com/ctralie/128cc07da67f1d2e10ea470ee2d23fe8)

    Parameters
    ----------
    D : ndarray (N, N) 
        An NxN distance matrix for points
    Return
    ------
    tuple (list, list) 
        (permutation (N-length array of indices), 
        lambdas (N-length array of insertion radii))
    """
    
    N = DM.shape[0]
    
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = DM[0, :]
    
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, DM[idx, :])
    
    return (perm, lambdas)

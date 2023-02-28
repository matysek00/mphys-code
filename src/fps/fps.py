
import numpy as np
import sparse
from scipy.spatial.distance import pdist, squareform
from distancecribe.descriptors import SOAP


def calculate_soap(
    data,
    species: list = ["H", "C"],
    r_cut: float = 6.0,
    n_max: int = 8,
    l_max: int = 8,
    periodic: bool = True,
    average: str = "inner",
    sparse: bool = True,
    ):
    """calculate soap data, very redundant, only changed the default values
    """

    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        periodic=periodic,
        average=average,
        sparse = sparse)

    soap_data = soap.create(data)
    return soap_data


def get_DM(file: str, test_size: int) -> np.array:
    """create distance matrix from soap data
    
    Parameters:
    ----------
    file: str
      .npz file containing soap data
    test_size: int
        how many of the last trajectories should be ignored

    Returns:
    --------
    DM: np.array
        distance matrix
    """
    
    soap_data = load_soap(file, test_size)    
    DM = Distance_matrix(soap_data)
    return DM


def Distance_matrix(data: np.array) -> np.array:
    """np.sqrt(2)*squareform(pdist(data, 'cosine'))
    """
    # Use cosine distance acording to 
    # https://pubs.rsc.org/en/content/articlelanding/2016/CP/C6CP00415F
    
    DM = np.sqrt(2)*squareform(pdist(data, 'cosine'))
    return DM


def load_soap(file: str, test_size: int) -> np.array:
    """load soap data from a file
    
    Parameters:
    ----------
    file: str 
        .npz file containing soap data
    test_size: int
        how many of the last trajectories should be ignored   

    Returns:
    --------
    dense: np.array
        soap data
    """

    soap_data = sparse.load_npz(file)

    if test_size == 0:
        dense = soap_data.todense()
    else:
        dense = soap_data.todense()[:-test_size]

    return dense 


def getGreedyPerm(DM: np.array) -> tuple(np.array, np.array):
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
    perm : ndarray (N,)
        The permutation of points
    lambdas : ndarray (N,)
        the distance between each point and the previous point
    """
    
    N = DM.shape[0]
    
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    
    # distances from the fisrt point to all other points
    distance = DM[0, :]

    
    for i in range(1, N):
        # furthest point 
        idx = np.argmax(distance)
        perm[i] = idx
        lambdas[i] = distance[idx]

        # update the distances so each point shows the distance
        # to the nearest point in the current set
        distance = np.minimum(distance, DM[idx, :])
    
    return (perm, lambdas)

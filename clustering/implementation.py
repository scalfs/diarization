import numpy as np
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.cluster import KMeans
from scipy.ndimage.filters import gaussian_filter

random_state = 222


def spectral_clustering(data):
    # # Spectral clustering
    # cossine similarity
    similarity = np.dot(data, data.T)
    # squared magnitude of preference vectors (number of occurrences) (diagonals are ai*ai)
    square_mag = np.diag(similarity)
    # inverse squared magnitude
    inv_square_mag = 1 / square_mag
    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)
    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    A = cosine.T * inv_mag
    # Fill the diagonals with very large negative value
    np.fill_diagonal(A, -1000)
    # Fill the diagonals with the max of each row
    np.fill_diagonal(A, A.max(axis=1))
    # final step in cossine sim
    A = (1-A)/2
    # Gaussian blur
    sigma = 0.5  # we will select sigma as 0.5
    A_gau = gaussian_filter(A, sigma)
    # Thresholding using multiplier = 0.01
    threshold_multiplier = 0.01
    A_thresh = A_gau * threshold_multiplier
    # Symmetrization
    A_sym = np.maximum(A_thresh, A_thresh.T)
    # Diffusion
    A_diffusion = A_sym * A_sym.T
    # Row-wise matrix Normalization
    Row_max = A_diffusion.max(axis=1).reshape(1, A_diffusion.shape[0])
    A_norm = A_diffusion / Row_max.T
    # Eigen decomposition
    eigval, eigvec = np.linalg.eig(A_norm)
    # Since eigen values cannot be negative for Positive semi definite matrix, the numpy returns negative values, converting it to positive
    eigval = np.abs(eigval)
    # reordering eigen values
    sorted_eigval_idx = np.argsort(eigval)[::-1]
    sorted_eigval = np.sort(eigval)[::-1]
    # For division according to the equation
    eigval_shifted = np.roll(sorted_eigval, -1)
    # Thresholding eigen values because we don't need very low eigan values due to errors
    eigval_thresh = 0.1
    sorted_eigval = sorted_eigval[sorted_eigval > eigval_thresh]
    eigval_shifted = eigval_shifted[:sorted_eigval.shape[0]]
    # Don't take the first value for calculations, if first value is large, following equation will return k=1, and we want more than one clusters
    # Get the argmax of the division, since its 0 indexed, add 1
    k = np.argmax(sorted_eigval[1:]/eigval_shifted[1:]) + 2
    print(f'Number of Eigen vectors to pick (clusters): {k}')
    # Get the indexes of eigen vectors
    idexes = sorted_eigval_idx[:k]
    A_eigvec = eigvec[:, idexes]
    A_eigvec = A_eigvec.astype('float32')

    # # K-Means offline clustering
    A_eigvec_norm = sk_normalize(A_eigvec)  # l2 normalized
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=random_state)
    kmeans.fit(A_eigvec)
    labels = kmeans.labels_

    return labels

import numpy as np
matrix = np.load("../data/data_clean/embed_matrix.npy")
for vec in matrix:
    print vec

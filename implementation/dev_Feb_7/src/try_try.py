import numpy as np
matrix = np.load("../data/data_clean/embed_matrix.npy")
count = 0
for vec in matrix:
    if count < 50:
        print vec
        count += 1

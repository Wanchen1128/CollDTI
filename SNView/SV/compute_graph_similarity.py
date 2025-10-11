import os
import gc
import time
import numpy as np
from scipy.spatial.distance import jaccard

Nets = ['mat_drug_drug', 'mat_drug_disease', 'mat_drug_se',
        'mat_protein_protein', 'mat_protein_disease']

os.makedirs('SNView/SV/network', exist_ok=True)

def jaccard_similarity(a, b):
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 - intersection / union if union != 0 else 0.0

def compute_sim_chunked(M, chunk_size=1000):
    n = M.shape[0]
    Sim = np.zeros((n, n), dtype=np.float32)
    
    np.fill_diagonal(Sim, 1.0)
    
    for i in range(0, n, chunk_size):
        i_end = min(i + chunk_size, n)
        for j in range(i+1, n, chunk_size):
            j_end = min(j + chunk_size, n)
            
            block = np.zeros((i_end-i, j_end-j), dtype=np.float32)
            for idx in range(i, i_end):
                for jdx in range(j, j_end):
                    if idx != jdx:
                        block[idx-i, jdx-j] = 1 - jaccard(M[idx], M[jdx])
            
            Sim[i:i_end, j:j_end] = block
            Sim[j:j_end, i:i_end] = block.T
            
            del block
            gc.collect()
            
    return Sim

for net in Nets:
    start_time = time.time()
    input_path = f'data/Ning/{net}.txt'
    output_path = f'SNView/SV/network/Sim_{net}.txt'
    
    M = np.loadtxt(input_path)
    M = (M > 0).astype(np.uint8)
    
    print(f"Processing {net} with shape {M.shape}...")
    
    chunk_size = max(500, min(2000, M.shape[0] // 10))
    
    Sim = compute_sim_chunked(M, chunk_size)

    Sim = np.nan_to_num(Sim, nan=0.0)

    np.savetxt(output_path, Sim, delimiter='\t', fmt='%.6f')

    print(f"Completed {net} in {time.time()-start_time:.2f} seconds")
    del M, Sim
    gc.collect()

M_drug = np.loadtxt('data/Ning/Similarity_Matrix_Drugs.txt')
np.savetxt('SNView/SV/network/Sim_mat_Drugs.txt', M_drug, delimiter='\t', fmt='%.6f')

M_protein = np.loadtxt('data/Ning/Similarity_Matrix_Proteins.txt')
np.savetxt('SNView/SV/network/Sim_mat_Proteins.txt', M_protein, delimiter='\t', fmt='%.6f')



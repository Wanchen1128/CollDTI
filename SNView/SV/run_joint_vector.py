import numpy as np
import time
import os

def diffusionRWR(A, maxiter, restartProb):
    n = A.shape[0]
    
    col_sum = np.sum(A, axis=0)
    isolated_nodes = (col_sum == 0)
    A = A + np.diag(isolated_nodes.astype(int))
    
    col_sum = np.sum(A, axis=0)
    P = A / col_sum
    
    restart = np.eye(n)
    Q = np.eye(n)
    
    for i in range(maxiter):
        Q_new = (1 - restartProb) * P @ Q + restartProb * restart
        delta = np.linalg.norm(Q - Q_new, 'fro')
        Q = Q_new
        if delta < 1e-6:
            break
    
    return Q

def joint(networks, rsp, maxiter):
    Q_list = []
    for net_file in networks:
        file_path = f'SNView/SV/network/{net_file}.txt'
        net = np.loadtxt(file_path)
        tQ = diffusionRWR(net, maxiter, rsp)
        Q_list.append(tQ)
    
    Q_combined = np.hstack(Q_list)
    
    nnode = Q_combined.shape[0]
    alpha = 1 / nnode
    Q_transformed = np.log(Q_combined + alpha) - np.log(alpha)

    return Q_transformed

if __name__ == "__main__":
    maxiter = 20
    restartProb = 0.50

    drugNets = ['Sim_mat_drug_drug', 'Sim_mat_drug_disease', 'Sim_mat_drug_se', 'Sim_mat_Drugs']
    proteinNets = ['Sim_mat_protein_protein', 'Sim_mat_protein_disease', 'Sim_mat_Proteins']

    os.makedirs('SNView/SV/feature', exist_ok=True)

    start_time = time.time()
    X = joint(drugNets, restartProb, maxiter)
    print(f"Drug networks processed in {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    Y = joint(proteinNets, restartProb, maxiter)
    print(f"Protein networks processed in {time.time() - start_time:.2f} seconds")
    
    np.savetxt('SNView/SV/feature/drug_vector.txt', X, delimiter='\t', fmt='%.6f')
    np.savetxt('SNView/SV/feature/protein_vector.txt', Y, delimiter='\t', fmt='%.6f')


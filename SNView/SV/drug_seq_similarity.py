import os
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs

def save_similarity_matrix(matrix, filename, matrix_name="Similarity Matrix", format="%.4f"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    n = matrix.shape[0]
    
    with open(filename, 'w') as f:
        np.savetxt(f, matrix, delimiter='\t')
    
    print(f"The matrix has been saved to {filename}")


drugbank_ids = pd.read_csv("data/Ning/DrugSeq.csv", header=None)

id_to_smiles = {}
for i in range(len(drugbank_ids)):
    id_to_smiles[drugbank_ids.iloc[i,0]]=drugbank_ids.iloc[i,1]

fingerprints = []
for i in range(len(drugbank_ids)):
    db_id = drugbank_ids.iloc[i,0]
    if id_to_smiles[db_id] == "0":
        fingerprints.append("0")
    else:
        mol = Chem.MolFromSmiles(id_to_smiles[db_id])
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fingerprints.append(fp)

n = len(fingerprints)
similarity_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if fingerprints[i] == "0" or fingerprints[j] == "0":
            similarity_matrix[i][j] = 0
        else:
            similarity_matrix[i][j] = DataStructs.TanimotoSimilarity(
                fingerprints[i], fingerprints[j]
            )

print("Similarity Matrix:")
save_similarity_matrix(similarity_matrix, "data/Ning/Similarity_Matrix_Drugs.txt", matrix_name="Similarity Matrix", format="%.4f")

print(similarity_matrix)
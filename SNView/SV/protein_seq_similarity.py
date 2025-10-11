import os
import parasail
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasketch import MinHash
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations_with_replacement
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


def calculate_protein_similarity_parasail(seq_pair):
    seq1, seq2 = seq_pair
    result = parasail.nw_stats(seq1, seq2, 10, 1, parasail.blosum62)
    
    if result.length == 0:
        return 0.0
    
    return result.matches / result.length


def compute_similarity(args):
    i, j, seq1, seq2 = args
    if pd.isna(seq1) or pd.isna(seq2):
        return i, j, 0
    return i, j, calculate_protein_similarity_parasail((seq1, seq2))

def create_protein_similarity_matrix(protein_sequences):
    n = len(protein_sequences)
    similarity_matrix = np.zeros((n, n))
    tasks = []
    
    for i, j in combinations_with_replacement(range(n), 2):
        tasks.append((i, j, protein_sequences[i], protein_sequences[j]))
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(compute_similarity, tasks), total=len(tasks)))
    
    for i, j, sim in results:
        similarity_matrix[i, j] = sim
        similarity_matrix[j, i] = sim
    
    return similarity_matrix


def create_protein_similarity_matrix_kmer(protein_sequences, k=3):
    valid_sequences = [seq if not pd.isna(seq) else "" for seq in protein_sequences]
    
    def get_kmers(seq, k):
        return [seq[i:i+k] for i in range(len(seq)-k+1)]
    
    vectorizer = CountVectorizer(analyzer=lambda s: get_kmers(s, k))
    kmer_matrix = vectorizer.fit_transform(valid_sequences)
    
    similarity_matrix = cosine_similarity(kmer_matrix)
    
    for i, seq in enumerate(protein_sequences):
        if pd.isna(seq):
            similarity_matrix[i, :] = 0
            similarity_matrix[:, i] = 0
    
    return similarity_matrix



def create_protein_similarity_matrix_minhash(protein_sequences, k=5, num_perm=128):
    n = len(protein_sequences)
    minhashes = []
    
    for seq in protein_sequences:
        if pd.isna(seq):
            minhashes.append(None)
            continue
            
        m = MinHash(num_perm=num_perm)
        for i in range(len(seq) - k + 1):
            m.update(seq[i:i+k].encode('utf-8'))
        minhashes.append(m)
    
    similarity_matrix = np.zeros((n, n))
    
    for i in tqdm(range(n)):
        for j in range(i, n):
            if minhashes[i] is None or minhashes[j] is None:
                sim = 0.0
            else:
                sim = minhashes[i].jaccard(minhashes[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
    
    return similarity_matrix


def save_similarity_matrix(matrix, filename, matrix_name="Similarity Matrix", format="%.4f"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    n = matrix.shape[0]
    
    with open(filename, 'w') as f:
        np.savetxt(f, matrix, fmt=format, delimiter='\t')
    
    print(f"The matrix has been saved to {filename}")

df = pd.read_csv("data/Ning/ProteinSeq.csv", header=None)
uniprot_ids = list(df.iloc[:,0])
protein_sequences = []
for i in range(len(df)):
    protein_sequences.append(df.iloc[i,1])

protein_sim_matrix = create_protein_similarity_matrix(protein_sequences)
print("Protein sequence similarity matrix:")
save_similarity_matrix(protein_sim_matrix, "data/Ning/Similarity_Matrix_Proteins.txt", matrix_name="Similarity Matrix", format="%.4f")

print("Save completed!")



import os
import dgl
import torch
import numpy as np
import pandas as pd
from operator import itemgetter
from SNView.SNmodel import SNView
from sklearn.model_selection import train_test_split, KFold

def load_dataset(network_path):
    """
    meta_path of drug
    """
    drug_drug = np.loadtxt(network_path + 'mat_drug_drug.txt')
    drug_disease = np.loadtxt(network_path + 'mat_drug_disease.txt')
    disease_drug = drug_disease.T
    drug_sideeffect = np.loadtxt(network_path + 'mat_drug_se.txt')
    sideeffect_drug = drug_sideeffect.T
    "-o"
    drug_drug_protein = np.loadtxt(network_path + 'mat_drug_protein_o.txt')
    """
    meta_path of protein
    """
    protein_protein = np.loadtxt(network_path + 'mat_protein_protein.txt')
    protein_protein_drug = drug_drug_protein.T
    protein_disease = np.loadtxt(network_path + 'mat_protein_disease.txt')
    disease_protein = protein_disease.T

    num_drug = drug_drug.shape[0]
    num_protein = protein_protein.shape[0]

    dg = dgl.heterograph({
        ('drug', 'dd', 'drug'): drug_drug.nonzero(),
        ('drug', 'dse', 'sideeffect'): drug_sideeffect.nonzero(),
        ('sideeffect', 'sed', 'drug'): sideeffect_drug.nonzero(),
        ('drug', 'ddi', 'disease'): drug_disease.nonzero(),
        ('disease', 'did', 'drug'): disease_drug.nonzero(),
        ('drug', 'ddp', 'protein'): drug_drug_protein.nonzero(),
        ('protein', 'pdd', 'drug'): protein_protein_drug.nonzero()
    })
    for etype in dg.etypes:
        src_type, edge_type, dst_type = dg.to_canonical_etype(etype)
        if src_type == dst_type:
            dg = dgl.add_self_loop(dg, etype=etype)

    pg = dgl.heterograph({
        ('protein', 'pp', 'protein'): protein_protein.nonzero(),
        ('protein', 'pdi', 'disease'): protein_disease.nonzero(),
        ('disease', 'dip', 'protein'): disease_protein.nonzero(),
        ('protein', 'pdd', 'drug'): protein_protein_drug.nonzero(),
        ('drug', 'ddp', 'protein'): drug_drug_protein.nonzero()

    })
    for etype in pg.etypes:
        src_type, edge_type, dst_type = pg.to_canonical_etype(etype)
        if src_type == dst_type:
            pg = dgl.add_self_loop(pg, etype=etype)

    graph = [dg, pg]
    node_num = [num_drug, num_protein]
    all_meta_paths = [[['dd'], ['dse', 'sed'], ['ddi', 'did'], ['ddp', 'pdd']],
                      [['pp'], ['pdi', 'dip'], ['pdd', 'ddp']]]

    return graph, node_num, all_meta_paths

def get_clGraph(data, task):
    cledg = np.loadtxt(f"{task}_cledge.txt", dtype=int)
    cl = torch.eye(len(data))
    for i in cledg:
        cl[i[0]][i[1]] = 1
    return cl

def get_cross(all_data, split=5):
    """
    :param data: dataset and label
    :return:
    testset index and trainset index
    """
    set1 = []
    set2 = []
    set3 = []
    skf = KFold(n_splits=split, shuffle=True)
    temp_features = np.ones((len(all_data),2))
    for train_val_index, test_index in skf.split(temp_features):
        getter = itemgetter(*train_val_index)
        train_val = list(getter(all_data))
        getter = itemgetter(*test_index)
        test = list(getter(all_data))

        train, val = train_test_split(train_val, test_size=0.1)

        set1.append(train)
        set2.append(val)
        set3.append(test)

    return set1, set2, set3


def get_complete_data(drug_seq_path, pro_seq_path, dti_path, d_snv, p_snv, split=5):
    protein_seq = pd.read_csv(pro_seq_path, header=None)
    protein_seq.iloc[:, 1].fillna('a', inplace=True)
    no_protein_seq_ind = []
    for i in range(len(protein_seq)):
        if protein_seq.iloc[i, 1] == 'a':
            no_protein_seq_ind.append(i)

    drug_smiles = pd.read_csv(drug_seq_path, header=None)
    no_drug_seq_ind = []
    for i in range(len(drug_smiles)):
        if drug_smiles.iloc[i, 1] == '0' or len(drug_smiles.iloc[i, 1]) > 250:
            no_drug_seq_ind.append(i)

    drug_ids_all = drug_smiles.iloc[:, 0].astype(str)
    protein_ids_all = protein_seq.iloc[:, 0].astype(str)

    dti_o = np.loadtxt(dti_path)
    train_positive_index = []
    whole_negative_index = []

    for i in range(np.shape(dti_o)[0]):
        if i not in no_drug_seq_ind:
            for j in range(np.shape(dti_o)[1]):
                if j not in no_protein_seq_ind:
                    if int(dti_o[i][j]) == 1:
                        train_positive_index.append([i, j])
                    else:
                        whole_negative_index.append([i, j])

    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=len(train_positive_index), replace=False)
    data_set = np.zeros((len(negative_sample_index) + len(train_positive_index), 3), dtype=int)
    count = 0

    for i in train_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1

    for i in range(len(negative_sample_index)):
        data_set[count][0] = whole_negative_index[negative_sample_index[i]][0]
        data_set[count][1] = whole_negative_index[negative_sample_index[i]][1]
        data_set[count][2] = 0
        count += 1

    all_data = []
    for i in range(data_set.shape[0]):
        drug_idx = data_set[i][0]
        prot_idx = data_set[i][1]
        label    = data_set[i][2]

        all_data.append([
            drug_smiles.iloc[drug_idx, 1],
            protein_seq.iloc[prot_idx, 1],
            d_snv[drug_idx],
            p_snv[prot_idx],
            drug_ids_all.iloc[drug_idx],
            protein_ids_all.iloc[prot_idx],
            label
        ])

    train_data, val_data, test_data = get_cross(all_data, split=split)

    return train_data, val_data, test_data


def process_data(data_root_path="./data/Ning/", sv_path="SNView/SV/feature/", device="cuda:0"):
    # load graph data
    graph, num, all_meta_paths = load_dataset(data_root_path)
    print("num of drug and protein:", num)
    graph = [g for g in graph]

    num_protein = num[1]
    num_drug = num[0]
    in_size = 512
    hidden_size = 256
    out_size = 128
    dropout = 0.5

    hd = torch.randn((num_drug, in_size))
    hp = torch.randn((num_protein, in_size))
    features_d = hd
    features_p = hp
    node_feature = [features_d, features_p]
    
    # Similarity View
    d_sv = torch.tensor(np.loadtxt(os.path.join(sv_path, 'drug_dae.txt')))
    p_sv = torch.tensor(np.loadtxt(os.path.join(sv_path, 'protein_dae.txt')))
    # Neighbor View
    model = SNView(all_meta_paths=all_meta_paths, in_size=[in_size,in_size],
                   hidden_size=[hidden_size, hidden_size],
                   hidden_size1=out_size,out_size=[out_size,out_size],
                   dropout=dropout)
    complete_model = torch.load("SNView/NV.pth")
    part_model_params = {}
    for name, param in complete_model.items():
        if name in model.state_dict():
            part_model_params[name] = param
    model.load_state_dict(part_model_params, strict=True)
    model.eval()
    with torch.no_grad():
        d_snv, p_snv = model.forward(graph=graph, h=node_feature, d_sv=d_sv, p_sv=p_sv)

    drug_seq_path = "./data/Ning/DrugSeq.csv"
    pro_seq_path = "./data/Ning/ProteinSeq.csv"
    dti_path = "./data/Ning/mat_drug_protein.txt"
    train_data, val_data, test_data = get_complete_data(drug_seq_path=drug_seq_path,
                                                        pro_seq_path=pro_seq_path,
                                                        dti_path=dti_path,
                                                        d_snv=d_snv, p_snv=p_snv)

    return train_data, val_data, test_data

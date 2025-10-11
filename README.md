## CollDTI
CollDTI: A Collaborative Multi-view Learning Framework for Drug-Target Interaction Prediction.
### Dependencies
- Python==3.8.19
- PyTorch==2.0.0+cu118
- DGL==1.1.2+cu118
- Numpy==1.24.2
- Panda==2.0.3
- RDKit==2024.3.5
### Quick start
- Run `python main.py`   You can recurrence our model.
### Code and data
The file structure is as follows: 
```
CollDTI
└───configs
│   └─── CollDTI.yaml
└───data
│   └─── Ning
│   └─── Luo
└───SNView
│   └─── SV
│   │    └─── au_class.py
│   │    └─── compute_graph_similarity.py
│   │    └─── DAE.py
│   │    └─── drug_seq_similarity.py
│   │    └─── protein_seq_similarity.py
│   │    └─── run_DAE.py
│   │    └─── run_DAE.py
│   │    └─── run_joint_vector.py
│   └─── SNmodel.py
└───dataloader.py
└───main.py
└───modules.py
└───process_data.py
└───trainer.py
└───utils.py

```

#### `data/`directory
##### `Luo/`
The Luo's dataset mentioned in the text can be found in [https://github.com/luoyunan/DTINet](https://github.com/luoyunan/DTINet).
- `drug_id.txt`: list of drug ids
- `protein_id.txt`: list of protein idss
- `disease_name.txt`: list of disease names
- `side-effect-name.txt`: list of side effect names
- `mat_drug_se.txt` : Drug-SideEffect association matrix
- `mat_protein_protein.txt` : Protein-Protein interaction matrix
- `mat_drug_protein.txt`: Drug_Protein interaction matrix
- `mat_drug_drug.txt`: Drug-Drug interaction matrix
- `mat_protein_disease.txt` : Protein-Disease association matrix
- `mat_drug_disease.txt`: Drug-Disease association matrix
Supplementary data:
- `DrugSeq.csv`: SMILES sequence corresponding to drug DrugBank ID (supplementary data)
- `ProteinSeq.csv`: sequence corresponding to protein UniProt ID (supplementary data)
##### `Ning/`
The Ning's dataset mentioned in the text can be found in [https://github.com/ningq669/DMHGNN](https://github.com/ningq669/DMHGNN).
- `drug_id.txt`: list of drug ids
- `protein_id.txt`: list of protein idss
- `disease_name.txt`: list of disease names
- `side-effect-name.txt`: list of side effect names
- `mat_drug_se.txt` : Drug-SideEffect association matrix
- `mat_protein_protein.txt` : Protein-Protein interaction matrix
- `mat_drug_protein.txt`: Drug_Protein interaction matrix
- `mat_drug_drug.txt`: Drug-Drug interaction matrix
- `mat_protein_disease.txt` : Protein-Disease association matrix
- `mat_drug_disease.txt`: Drug-Disease association matrix
- `DrugSeq.csv`: SMILES sequence corresponding to drug DrugBank ID (supplementary data)
- `ProteinSeq.csv`: sequence corresponding to protein UniProt ID (supplementary data)

### Contacts
Dear editors and reviewers, please feel free to contact us if there are any issues during the software execution！
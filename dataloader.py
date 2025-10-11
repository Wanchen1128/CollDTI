import torch
import torch.utils.data as data
from functools import partial
from utils import integer_label_protein
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer

class LoadDataset(data.Dataset):
    def __init__(self, list_IDs, df, max_drug_nodes=290):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        drug = self.df[index][0]
        drug = self.fc(smiles=drug, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = drug.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        drug.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        drug.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        drug = drug.add_self_loop()

        pro = self.df[index][1]
        pro = integer_label_protein(pro)

        y = self.df[index][-1]

        d_snv = self.df[index][2]
        p_snv = self.df[index][3]

        return drug, pro, d_snv, p_snv, y



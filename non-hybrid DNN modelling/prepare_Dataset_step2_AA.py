import math
import os
import pickle
import subprocess
import torch
import os.path as osp
from torch_geometric.data import Dataset, download_url, HeteroData
from biopandas.pdb import PandasPdb
import Constants
from Dataset.distanceTransform import myDistanceTransform
from Dataset.myKnn import myKNNGraph
from Dataset.myRadiusGraph import myRadiusGraph
from Dataset.utils import find_files, process_pdbpandas, get_knn, generate_Identity_Matrix
import torch_geometric.transforms as T
from torch_geometric.data import Data
from Dataset.AdjacencyTransform import AdjacencyFeatures
from preprocessing.utils import pickle_load, pickle_save, get_sequence_from_pdb, fasta_to_dictionary, collect_test, \
    read_test_set, read_test, cafa_fasta_to_dictionary
import pandas as pd
import random
from Dataset.cla1 import embedding_construction
from Dataset.cla2 import pos_chain_pos

class PDBDataset(Dataset):

    def __init__(self, root, pdb_p, esm_p, formdata_p, forminfo, transform=None, pre_transform = None, pre_filter=None):

        self.root = root
        self.pre_transform = pre_transform
        
        # formulation data          
        form = pd.read_csv(formdata_p)
        self.prot_ids = form['Protein'].tolist()
     
        self.pdb_pth = pdb_p
        self.esm_path = esm_p
        self.forminfo = forminfo
        
        self.processed_file_list = []

        self.data = self.prot_ids
        
        print(f'pre_transform in __init__: {self.pre_transform}')
        for i in self.data:

            self.processed_file_list.append('{}.pt'.format(i)) 
        
        

        super().__init__(self.root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return self.pdb_pth

    @property
    def processed_dir(self) -> str:
        return self.root

    @property
    def processed_file_names(self):
        return self.processed_file_list


    def process(self):
         if self.pre_transform is None:
            raise ValueError("pre_transform is None, check initialization logic")

         print(f'pre_transform in process: {self.pre_transform}')
        
        
         rem_files = set(self.processed_file_list)

         print("{} unprocessed proteins out of {}".format(len(rem_files), len(self.processed_file_list)))
         #chain_id = 'A'

         for file in rem_files:

             protein = file.rsplit("_", 1)[0]
             protein_form = file.split(".")[0]
             
             #print("Processing protein {} for {}".format(protein, protein_form))

             raw_path = self.raw_dir + '{}.pdb.gz'.format(protein)

             emb = torch.load(self.root + "/esm/{}/{}.pt".format(self.esm_path, protein), weights_only = False)
             embedding_features_per_residue = emb
             
             #print('protein_form', protein_form)
             #print('embedding_features_per_residue', embedding_features_per_residue.shape)

             form = pd.read_csv(formdata_p)
             
             labels_df = form.iloc[:, 21: self.forminfo] # collidial, -3; conformation, -2; solubility, -4; viscosity, -5             
             #print('labels_df',labels_df)
             labels_list = labels_df.values.tolist()
             labels_dict = dict(zip(form['Protein'], labels_list))               
             form_one = torch.tensor(labels_dict[protein_form], dtype=torch.float32).view(1,-1)

             AA_df = form.iloc[:, 1: 21] # collidial, -3; conformation, -2; solubility, -4; viscosity, -5             
             AA_list = AA_df.values.tolist()
             AA_dict = dict(zip(form['Protein'], AA_list))               
             AA = torch.tensor(AA_dict[protein_form], dtype=torch.float32).view(1,-1)

             if raw_path:

                 node_coords, chain_coordinates = pos_chain_pos(raw_path)
                 
             node_size = node_coords.shape[0]
             data = HeteroData()

             data['atoms'].pos = node_coords
             data['atoms'].embedding_features_per_residue = embedding_features_per_residue            
             data['atoms'].form_one = form_one
             data['atoms'].AA = AA             
             data['atoms'].protein = protein_form
             data['atoms'].chainpos = chain_coordinates
             #print("data['atoms'].protein",data['atoms'].protein)



             if self.pre_transform is not None:
                 _transforms = []
                 for i in self.pre_transform:
                     if i[0] == "KNN":
                         kwargs = {'mode': i[1], 'sequence_length': node_size}
                         knn = get_knn(**kwargs)
                         _transforms.append(myKNNGraph(i[1], k=knn, force_undirected=True))
                     if i[0] == "DIST":
                         _transforms.append(myRadiusGraph(i[1], r=i[2], loop=False))
                 #_transforms.append(myDistanceTransform(edge_types=self.pre_transform, norm=True))
                 _transforms.append(AdjacencyFeatures(edge_types=self.pre_transform))
                 
                 #print('pre_transform1', _transforms)
                 pre_transform = T.Compose(_transforms)

                 data = pre_transform(data)
                 #print('data', data)

             torch.save(data, osp.join(self.root + "/processed_data/{}/{}.pt".format(self.esm_path, protein_form)))


    def len(self):
        return len(self.data)

    def get(self, idx):
    
        train = pd.read_csv(formdata_p)
        labels = train.set_index('Protein')['Label'].to_dict()
        
        rep = self.data[idx]
        label = labels[rep]
        
        return torch.load(osp.join(self.root + "/processed_data/{}/{}.pt".format(self.esm_path, protein_form)),  weights_only=False), label 

def load_dataset(root, pdb_p, esm_p, formdata_p, forminfo):

    if root == None:
        raise ValueError('Root path is empty, specify root directory')
       
    pre_transform = [("KNN", "cbrt", "cbrt", "K nearest neighbour with sqrt for neighbours")]

    #embedding_construction(**kwargs)
   # print('pre_transform', pre_transform)
    dataset = PDBDataset(root, pdb_p, esm_p, formdata_p, forminfo, pre_transform=pre_transform)
    
    return dataset

mapping = {
    "0_conformation": {"csv": "conformation_merged_proteins_class.csv", "formcol": -3},
    "0_solubility": {"csv": "solubility_merged_proteins_20250425.csv", "formcol": -4},
    "0_collodial": {"csv": "collidial_merged_proteins.csv", "formcol": -3},
    "0_viscosity": {"csv": "viscocity_merged_proteins.csv", "formcol": -5},
    "0_solubility_class": {"csv": "solubility_merged_proteins_20250522.csv", "formcol": -5},
    
    "exp_kd":{"csv": "experiment/kd_2025118.csv", "formcol": -2},
    "exp_tm":{"csv": "experiment/Tm_2025118_class.csv", "formcol": -2},
    "exp_viscosity":{"csv": "experiment/viscosity_2025118.csv", "formcol": -3},
    "exp_solubility":{"csv": "experiment/solubility_20250425.csv", "formcol": -3},
    "exp_solubility_class":{"csv": "experiment/solubility_20250522.csv", "formcol": -4},       
     
    "0_conformation_AA": {"csv": "conformation_merged_proteins_class_AA.csv", "formcol": -3},
    "0_solubility_AA": {"csv": "solubility_merged_proteins_20250425_AA.csv", "formcol": -4},
    "0_collodial_AA": {"csv": "collidial_merged_proteins_AA.csv", "formcol": -3},
    "0_viscosity_AA": {"csv": "viscocity_merged_proteins_AA.csv", "formcol": -5},
    "0_solubility_class_AA": {"csv": "solubility_merged_proteins_20250522_AA.csv", "formcol": -5},
    
    "exp_kd_AA":{"csv": "experiment/kd_2025118_AA.csv", "formcol": -2},
    "exp_tm_AA":{"csv": "experiment/Tm_2025118_class_AA.csv", "formcol": -2},
    "exp_viscosity_AA":{"csv": "experiment/viscosity_2025118_AA.csv", "formcol": -3},
    "exp_solubility_AA":{"csv": "experiment/solubility_20250425_AA.csv", "formcol": -3},  
    "exp_solubility_class_AA":{"csv": "experiment/solubility_20250522_AA.csv", "formcol": -4},       
}



esm_p = "0_solubility_class_AA"
pdb_p = f"/media/ouyang/backup_disk/output/0data/pdb/gzpdb/{esm_p}_gz/"
root = "/media/ouyang/backup_disk/output/0data"

#esm_p = "exp_solubility_class_AA"
#pdb_p = "/media/ouyang/backup_disk/output/0data/pdb/gzpdb/experiment_gz/"
formadd = mapping[esm_p]["csv"]
formdata_p = f"/media/ouyang/backup_disk/output/0data/formdata/{formadd}"

forminfo = mapping[esm_p]["formcol"]

load_dataset(root, pdb_p, esm_p, formdata_p, forminfo)
print(f'finish for {esm_p}')








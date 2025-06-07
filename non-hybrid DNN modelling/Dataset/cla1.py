import os
from biopandas.pdb import PandasPdb
from Bio import SeqIO
from preprocessing.utils import pickle_save, pickle_load, save_ckp, load_ckp, class_distribution_counter, \
    draw_architecture, compute_roc, get_sequence_from_pdb,create_seqrecord, get_proteins_from_fasta, generate_bulk_embedding, fasta_to_dictionary
from Constants import INVALID_ACIDS, amino_acids
import torch

def get_sequences_from_pdb(pdb_file):
    """Extract sequences for all chains in a PDB file."""
    pdb_to_pandas = PandasPdb().read_pdb(pdb_file)
    pdb_df = pdb_to_pandas.df['ATOM']
    pdb_df = pdb_df[pdb_df['atom_name'] == 'CA']

    # Group by chain_id and extract sequences
    chain_sequences = {}
    for chain_id, group in pdb_df.groupby('chain_id'):
        residues = group['residue_name'].to_list()
        try:
            residues = ''.join([amino_acids[res] for res in residues])
            chain_sequences[chain_id] = residues
        except KeyError as e:
            print(f"Error: Unrecognized residue {e} in {pdb_file} (Chain {chain_id})")
            continue

    return chain_sequences


def create_fasta(proteins, pdb_p):
    """Generate a FASTA file for each chain in multi-chain proteins."""
    fasta = []
    for protein in proteins:
        pdb_file = os.path.join(pdb_p, f"{protein}.pdb.gz")
        if not os.path.exists(pdb_file):
            print(f"Error: PDB file {pdb_file} not found.")
            continue

        chain_sequences = get_sequences_from_pdb(pdb_file)
        for chain_id, sequence in chain_sequences.items():
            # Use '|' as a separator to avoid conflicts with existing '_'
            record_id = f"{protein}-{chain_id}"
            fasta.append(create_seqrecord(id=record_id, seq=sequence))

    fasta_path = os.path.join(pdb_p, "sequence.fasta")
    SeqIO.write(fasta, fasta_path, "fasta")
    return fasta_path


def generate_embeddings(fasta_path, output_dir):
    """
    Generate embeddings for each chain in the input fasta file,
    and combine embeddings for chains belonging to the same protein.
    """
    # Parse input sequences
    input_seq_iterator = SeqIO.parse(fasta_path, "fasta")
    sequences = list(input_seq_iterator)

    # Directory to store individual chain embeddings
    chain_embedding_dir = os.path.join(output_dir, "chains")
    os.makedirs(chain_embedding_dir, exist_ok=True)

    # Generate embeddings for each chain
    generate_bulk_embedding(
        path_to_extract_file="./preprocessing/extract.py",
        fasta_file=fasta_path,
        output_dir=chain_embedding_dir,
    )

    # Combine embeddings for chains of the same protein
    combined_embeddings = {}
    for seq_record in sequences:
        # Split by '|' to handle protein and chain separation
        protein_id, chain_id = seq_record.id.split("-", 1)
        embedding_file = os.path.join(chain_embedding_dir, f"{seq_record.id}.pt")
        #print('protein_id',protein_id)
        #print('chain_id',chain_id)
        if protein_id not in combined_embeddings:
            combined_embeddings[protein_id] = {
                "embeddings": [],
                "chain_ids": []
            }

        if os.path.exists(embedding_file):
            all_embedding = torch.load(embedding_file)
            embedding = all_embedding['representations'][33]  # Load chain embedding
            num_residues, embedding_dim = embedding.shape

            ## Add chain-specific feature (order based on sequence)
            chain_order = len(combined_embeddings[protein_id]["chain_ids"])  # Order of current chain
            combined_embeddings[protein_id]["chain_ids"].append(chain_id)

            chain_feature = torch.full((num_residues, 1), chain_order, dtype=torch.float32)  # Add order number
            embedding = torch.cat([embedding, chain_feature], dim=1)

            combined_embeddings[protein_id]["embeddings"].append(embedding)
        else:
            print(f"Warning: Embedding file for {seq_record.id} not found.")

    # Concatenate chain embeddings and save


    for protein_id, data in combined_embeddings.items():
        concatenated_embedding = torch.cat(data["embeddings"], dim=0)  # L × (D+1)
        torch.save(concatenated_embedding, os.path.join(output_dir, f"{protein_id}.pt"))

    print(f"Combined embeddings with chain feature saved to {output_dir}.")
    return output_dir


def embedding_construction(data_p, pdb_p, esm_p, input_type='pdb'):
    """Main function to construct embeddings for proteins."""
           
    #pdb_p = kwargs.get('pdb_path', None)
    #esm_p = kwargs.get('esm_path', None) 
    #data_p = 'data/esm'
    
    if input_type == 'fasta' and args.fasta_path:
        proteins = set(get_proteins_from_fasta(args.fasta_path))
        pdbs = set([i.split(".")[0] for i in os.listdir(pdb_p)])
        proteins = list(pdbs.intersection(proteins))
    elif input_type == 'pdb' and pdb_p:
        if os.path.exists(pdb_p):
            proteins = [
                protein.split('.')[0] for protein in os.listdir(pdb_p) if protein.endswith(".pdb.gz")
            ]
            if not proteins:
                print(f"No proteins found in {pdb_p}.")
                return
            fasta_path = create_fasta(proteins, pdb_p)
        else:
            #print(f"PDB directory not found: {pdb_p}")
            print('hello')
            return
    else:
        print("Invalid input type or missing paths.")
        return

    if proteins:
        print(f"Predicting for {len(proteins)} proteins")
        print(f"Generating Embeddings from {fasta_path}")
        os.makedirs(os.path.join(data_p, esm_p), exist_ok=True)
        generate_embeddings(fasta_path, os.path.join(data_p, esm_p))

pdb_p = '/media/ouyang/backup_disk/output/0allPDB/0_collodial'
esm_p = '0_collodial'
data_p = '/media/ouyang/backup_disk/output/0allPDB/'   
embedding_construction(data_p, pdb_p, esm_p)




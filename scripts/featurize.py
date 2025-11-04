import pandas as pd
import argparse
import numpy as np
from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer
from tqdm import tqdm
import traceback

def featurize_smiles(input_path, smiles_column, output_path):
    try:
        print(f"Loading input CSV: {input_path}")
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} rows from input CSV.")
        
        ckpt = "checkpoints/last.ckpt"
        print(f"Initializing MolBertFeaturizer with checkpoint: {ckpt}")
        featurizer = MolBertFeaturizer(
            checkpoint_path=ckpt,
            embedding_type="pooled",
            max_seq_len=200,
            device="cpu"
        )
        print("Featurizer initialized.")
        
        smiles = df[smiles_column].tolist()
        print(f"Extracted {len(smiles)} SMILES strings.")
        
        print("Starting featurization with progress bar...")
        vectors = []
        valid_mask = []
        for smi in tqdm(smiles, desc="Processing molecules", unit="molecule"):
            try:
                vec, valid = featurizer.transform([smi])
                vectors.append(vec[0])
                valid_mask.append(valid[0])
            except Exception as e:
                vectors.append(np.zeros(768))  # or handle error as needed
                valid_mask.append(False)
        print("Featurization completed.")
        
        num_valid = np.sum(valid_mask) if isinstance(valid_mask, np.ndarray) else sum(valid_mask)
        print(f"Number of valid SMILES: {num_valid}")
        print(f"Processed {num_valid} molecules out of {len(smiles)} total.")
        
        valid_df = df[valid_mask].reset_index(drop=True)
        
        if num_valid == 0:
            print("No valid SMILES found. Creating output with SMILES column and empty vector columns.")
            vec_df = pd.DataFrame(columns=[f'molbert_{i+1}' for i in range(768)])
            out_df = pd.concat([valid_df[[smiles_column]], vec_df], axis=1)
        else:
            print(f"Vectors shape: {np.array(vectors).shape if hasattr(vectors, 'shape') else 'Not ndarray'}")
            if not isinstance(vectors, np.ndarray):
                vectors = np.array(vectors)
            if vectors.size == 0:
                vectors = np.empty((0, 768))
            elif vectors.ndim == 1:
                vectors = vectors.reshape(num_valid, -1)
            vec_df = pd.DataFrame(vectors)
            vec_df.columns = [f'molbert_{i+1}' for i in range(vec_df.shape[1])]
            out_df = pd.concat([valid_df[[smiles_column]], vec_df], axis=1)
        
        print(f"Saving output to: {output_path}")
        out_df.to_csv(output_path, index=False)
        print("Output saved successfully.")
        print(f"Total molecules processed and saved: {len(out_df)}")
    
    except Exception as e:
        print("An error occurred:")
        print(traceback.format_exc())
        print("Failed to process and save the file.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Featurize SMILES from a CSV file and save embeddings.')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--smiles-column', type=str, required=True, help='Name of the SMILES column in input file')
    parser.add_argument('--output', type=str, required=True, help='Path to output CSV file with vectors')
    args = parser.parse_args()
    featurize_smiles(args.input, args.smiles_column, args.output)

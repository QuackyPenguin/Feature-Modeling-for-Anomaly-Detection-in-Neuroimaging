__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import argparse
import os
import pickle
import sys

import pandas as pd
import lmdb
import nibabel as nib
import numpy as np
from tqdm import tqdm

from normalize import * 


def main(args):
    split_and_save(args)


def split_and_save(args):
    print("Save and Split function in the file create_LMDB.py")
    # print the arguments
    print(args)
    print('Start creating LMDB files')
    
    if args.data_type not in ["T1w", "T2w"]:
        raise ValueError("Data type must be either T1w or T2w.")
    if args.dimension not in ["2d", "3d"]:
        raise ValueError("Dimension must be either 2d or 3d.")
    
    os.makedirs(f"{args.output}", exist_ok=True)
    os.makedirs(f"{args.output}/normal", exist_ok=True)
    os.makedirs(f"{args.output}/normal/{args.dimension}", exist_ok=True)
    
    df = pd.read_csv(args.data_set)
    pbar = tqdm(df['Paths'])
    map_size = 80000000 *len(pbar)        #  71424196 * len(pbar)
    env = lmdb.open(str(f"{args.output}/normal/{args.dimension}/{args.data_type}"), map_size=map_size)
    commit_threshold = {
        '2d': [75000, 75000],
        '3d': [500, 500]
    }
    
    try:
        txn = env.begin(write=True)
        num = 0
        for subject in pbar:
            img = np.asarray(nib.load(subject).dataobj,dtype=float)
            img = normalize_image(img)
            
            if args.dimension == "2d":
                for z in range(img.shape[2]):
                    slice = img[:, :, z]
                    if np.any(slice):
                        key = f"{num:08}"
                        num += 1
                        data = {"slice": slice, "depth": z}
                        txn.put(key.encode("ascii"), pickle.dumps(data))
                    
            elif args.dimension == "3d":
                # Transpose from (W, H, D) to (D, H, W)
                img = np.transpose(img, (2, 1, 0))
                
                key = f"{num:08}"
                num += 1
                txn.put(key.encode("ascii"), pickle.dumps(img))
                
            if num > commit_threshold[args.dimension][0]:
                tqdm.write(f"\nSaved {num} images and committing to LMDB...")
                sys.stdout.flush()
                txn.commit()
                txn = env.begin(write=True)
                
                commit_threshold[args.dimension][0] += commit_threshold[args.dimension][1]
                
            del img
        
        txn.commit()
        
    except Exception as e:
        print(e)
    finally:
        print("Closing environment")
        env.close()
    
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split the data into slices.")
    parser.add_argument(
        "-d",
        "--data_set",
        type=str,
        required=True,
        metavar="",
        help="The csv that contains the paths to the MRI-Volumes.",
    )
    parser.add_argument(
        "-t",
        "--data_type",
        type=str,
        default='T1w',
        required=True,
        metavar="",
        help="The BIDS ending for the MRI Sequence (T1w or T2w, etc).",
    )
    parser.add_argument(
        "-dim",
        "--dimension",
        type=str,
        default="3d",
        metavar="",
        help="Whether to save the slices (2d) or the entire volume (3d).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="/home/dkovacevic/dataset",
        metavar="",
        help="The path where the LMDB files are saved.",
    )
    args = parser.parse_args()
    main(args)

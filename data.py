import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import os
from pathlib import Path
from collections import defaultdict
import os
from argparse import Namespace
from joblib import dump, load
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader as dl
import random

base_dir = Path('/Users/vinay/Projects/Recsys')
if not base_dir:
    base_dir = Path(os.getcwd())


data_dir = base_dir/'data'/'archive'
store_dir = base_dir/'artifacts'



def reduce_sz(df,sz=10000):

    all_users = sorted(list(set(df['userId'])))
    data = []
    start = 0
    for user in tqdm(all_users):
        temp = df[df['userId'] == user]
        end = start+len(temp)
        temp = copy.deepcopy(temp.reindex(range(start,end)))
        start = end
        data.append(temp)
        if end > sz:
            break
    return pd.concat(data)


def set_seeds(seed=0):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # multi-GPU


def get_data(data_dir,split:list,mode,data_type,save_to_disk=False):
    #split = [0.6,0.5]
    #mode = 'random' or 'seq_aware'
    trn,vld,tst = defaultdict(list),defaultdict(list),defaultdict(list)
    df = pd.read_csv(data_dir/(data_type+'_'+'rating.csv'))
    #df = df.sample(frac=1,random_state=seed)
    perm = random.sample(range(len(df)),len(df))
    df = df.iloc[perm].reset_index()
    df['date'] = pd.to_datetime(df['timestamp'])
    user_ids = set(df['userId'])
    for user in tqdm(user_ids):
        tmp = df[df['userId'] == user]
        if split == 'seq_aware':
            tmp = tmp.sort_values(by='date')
        sz = len(tmp)
        #splitting-ids
        t,v = int(split[0]*sz),int(split[0]*sz)+int((sz-int(split[0]*sz))*split[-1])
        trn_ids,vld_ids,tst_ids = slice(0,t),slice(t,v),slice(v,sz)
        #train
        trn['user_id'].extend(tmp[trn_ids]['userId'].tolist())
        trn['rating'].extend(tmp[trn_ids]['rating'].tolist())
        trn['movie_id'].extend(tmp[trn_ids]['movieId'].tolist())
        #valid
        vld['user_id'].extend(tmp[vld_ids]['userId'].tolist())
        vld['rating'].extend(tmp[vld_ids]['rating'].tolist())
        vld['movie_id'].extend(tmp[vld_ids]['movieId'].tolist())
        #test
        tst['user_id'].extend(tmp[tst_ids]['userId'].tolist())
        tst['rating'].extend(tmp[tst_ids]['rating'].tolist())
        tst['movie_id'].extend(tmp[tst_ids]['movieId'].tolist())
    trn,vld,tst = pd.DataFrame(trn),pd.DataFrame(vld),pd.DataFrame(tst)
    if save_to_disk:
        trn.to_csv(data_dir/(data_type+'_'+mode+'_trn.csv'))
        vld.to_csv(data_dir/(data_type+'_'+mode+'_vld.csv'))
        tst.to_csv(data_dir/(data_type+'_'+mode+'_tst.csv'))
    return trn,vld,tst


class RecsysDataset(torch.utils.data.Dataset):
    def __init__(self,df,usr_dict=None,mov_dict=None):
        self.df = df
        self.usr_dict = usr_dict
        self.mov_dict = mov_dict

    def __getitem__(self,index):
        if self.usr_dict and self.mov_dict:
            return [self.usr_dict[int(self.df.iloc[index]['user_id'])],self.mov_dict[int(self.df.iloc[index]['movie_id'])]],self.df.iloc[index]['rating']
        else:
            return [int(self.df.iloc[index]['user_id']-1),int(self.df.iloc[index]['movie_id']-1)],self.df.iloc[index]['rating']

    def __len__(self):
        return len(self.df)


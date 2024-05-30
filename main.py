from DWLR.DWLR_model import DWLR
import os
import numpy as np
import argparse
import torch
import random
import pickle
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--adv_loss_weight', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--class_num', type=int, default=6)
parser.add_argument('--confidence', type=int, default=1.0)
parser.add_argument('--cuda', type=str, default=1)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--emb_dim', type=int, default=64)
parser.add_argument('--encoder', type=str, default='cnn')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--ff_bias', action='store_true')
parser.add_argument('--freq', action='store_true')
parser.add_argument('--IG', type=float, default=2)
parser.add_argument('--in_dim', type=int, default=3)
parser.add_argument('--label_regular', type=float, default=2.0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--l2', type=float, default=1e-3)
parser.add_argument('--n_freq', type=int, default=32)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_layer', type=int, default=4)
parser.add_argument('--run_times', type=int, default=3)
parser.add_argument('--patch_len', type=int, default=8)
parser.add_argument('--q_k_dim', type=int, default=32)
parser.add_argument('--reweight_weight', type=float, default=2.)
parser.add_argument('--seq_len', type=int, default=128)
parser.add_argument('--source', type=int, default=20)
parser.add_argument('--target', type=int, default=6)
parser.add_argument('--time', action='store_true')
parser.add_argument('--T', type=float, default=100)
parser.add_argument('--v_dim', type=int, default=32)


# use wisdm as example
def get_wisdm_data(user):
    data_path = './wisdm/dataset/'
    with open(os.path.join(data_path, f'{user}_datas.pkl'), 'rb') as fo:
        x = pickle.load(fo)
        x = (x-np.mean(x, axis=0))/(np.std(x, axis=0)+1e-8)
    with open(os.path.join(data_path, f'{user}_labels.pkl'), 'rb') as fo:
        y = pickle.load(fo)
    
    return x, y


def main(config, save_path):
    x_source, y_source = get_wisdm_data(config.source)
    x_target_all, y_target_all = get_wisdm_data(config.target)
    x_target, x_test, y_target, y_test = train_test_split(x_target_all, y_target_all, train_size=0.6, shuffle=True)
    model = DWLR(config, save_path)
    model.fit(x_source, y_source, x_target, config.epochs, y_target=y_target, x_test=x_test, y_test=y_test)

def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    set_global_random_seed(0)  # you can use different seed 
    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda)

    print(config.__dict__)
    main(config, save_path='./model_save')

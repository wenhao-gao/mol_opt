"""
This function is used to compute the mean reciprocal ranking for reactant 1 
selection using the different distance metrics in the k-NN search.
"""
from syn_net.models.mlp import MLP, load_array
from scipy import sparse
import numpy as np
from sklearn.neighbors import BallTree
import torch
from syn_net.utils.predict_utils import cosine_distance, ce_distance


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--featurize", type=str, default='fp',
                        help="Choose from ['fp', 'gin']")
    parser.add_argument("-r", "--rxn_template", type=str, default='hb',
                        help="Choose from ['hb', 'pis']")
    parser.add_argument("--param_dir", type=str, default='hb_fp_2_4096_256',
                        help="")
    parser.add_argument("--radius", type=int, default=2,
                        help="Radius for Morgan fingerprint.")
    parser.add_argument("--nbits", type=int, default=4096,
                        help="Number of Bits for Morgan fingerprint.")
    parser.add_argument("--out_dim", type=int, default=256,
                        help="Output dimension.")
    parser.add_argument("--ncpu", type=int, default=8,
                        help="Number of cpus")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="")
    parser.add_argument("--distance", type=str, default="euclidean",
                        help="Choose from ['euclidean', 'manhattan', 'chebyshev', 'cross_entropy', 'cosine']")
    args = parser.parse_args()

    if args.out_dim == 300:
        validation_option = 'nn_accuracy_gin'
    elif args.out_dim == 4096:
        validation_option = 'nn_accuracy_fp_4096'
    elif args.out_dim == 256:
        validation_option = 'nn_accuracy_fp_256'
    elif args.out_dim == 200:
        validation_option = 'nn_accuracy_rdkit2d'
    else:
        raise ValueError

    main_dir = '/pool001/whgao/data/synth_net/' + args.rxn_template + '_' + args.featurize + '_' + str(args.radius) + '_' + str(args.nbits) + '_' + validation_option[12:] + '/'
    path_to_rt1 = '/home/whgao/scGen/synth_net/synth_net/params/' + args.param_dir + '/' + 'rt1.ckpt'
    batch_size = args.batch_size
    ncpu = args.ncpu

    # X = sparse.load_npz(main_dir + 'X_rt1_train.npz')
    # y = sparse.load_npz(main_dir + 'y_rt1_train.npz')
    # X = torch.Tensor(X.A)
    # y = torch.Tensor(y.A)
    # _idx = np.random.choice(list(range(X.shape[0])), size=int(X.shape[0]/100), replace=False)
    # train_data_iter = load_array((X[_idx], y[_idx]), batch_size, ncpu=ncpu, is_train=False)

    # X = sparse.load_npz(main_dir + 'X_rt1_valid.npz')
    # y = sparse.load_npz(main_dir + 'y_rt1_valid.npz')
    # X = torch.Tensor(X.A)
    # y = torch.Tensor(y.A)
    # _idx = np.random.choice(list(range(X.shape[0])), size=int(X.shape[0]/10), replace=False)
    # valid_data_iter = load_array((X[_idx], y[_idx]), batch_size, ncpu=ncpu, is_train=False)

    X = sparse.load_npz(main_dir + 'X_rt1_test.npz')
    y = sparse.load_npz(main_dir + 'y_rt1_test.npz')
    X = torch.Tensor(X.A)
    y = torch.Tensor(y.A)
    _idx = np.random.choice(list(range(X.shape[0])), size=int(X.shape[0]/10), replace=False)
    test_data_iter = load_array((X[_idx], y[_idx]), batch_size, ncpu=ncpu, is_train=False)
    data_iter = test_data_iter

    rt1_net = MLP.load_from_checkpoint(path_to_rt1,
                    input_dim=int(3 * args.nbits),
                    output_dim=args.out_dim,
                    hidden_dim=1200,
                    num_layers=5,
                    dropout=0.5,
                    num_dropout_layers=1,
                    task='regression',
                    loss='mse',
                    valid_loss='mse',
                    optimizer='adam',
                    learning_rate=1e-4,
                    ncpu=ncpu)
    rt1_net.eval()
    rt1_net.to(args.device)

    bb_emb_fp_256 = np.load('/pool001/whgao/data/synth_net/st_hb/enamine_us_emb_fp_256.npy')

    # for kw_metric_ in ['euclidean', 'manhattan', 'chebyshev', 'cross_entropy', 'cosine']:
    kw_metric_ = args.distance

    if kw_metric_ == 'cross_entropy':
        kw_metric = ce_distance
    elif kw_metric_ == 'cosine':
        kw_metric = cosine_distance
    else:
        kw_metric = kw_metric_

    kdtree_fp_256 = BallTree(bb_emb_fp_256, metric=kw_metric)

    ranks = []
    for X, y in data_iter:
        X, y = X.to(args.device), y.to(args.device)
        y_hat = rt1_net(X)
        dist_true, ind_true = kdtree_fp_256.query(y.detach().cpu().numpy(), k=1)
        dist, ind = kdtree_fp_256.query(y_hat.detach().cpu().numpy(), k=bb_emb_fp_256.shape[0])
        ranks = ranks + [np.where(ind[i] == ind_true[i])[0][0] for i in range(len(ind_true))]

    ranks = np.array(ranks)
    rrs = 1 / (ranks + 1)
    np.save('ranks_' + kw_metric_ + '.npy', ranks)
    print(f"Result using metric: {kw_metric_}")
    print(f"The mean reciprocal ranking is: {rrs.mean():.3f}")
    print(f"The Top-1 recovery rate is: {sum(ranks < 1) / len(ranks) :.3f}, {sum(ranks < 1)} / {len(ranks)}")
    print(f"The Top-3 recovery rate is: {sum(ranks < 3) / len(ranks) :.3f}, {sum(ranks < 3)} / {len(ranks)}")
    print(f"The Top-5 recovery rate is: {sum(ranks < 5) / len(ranks) :.3f}, {sum(ranks < 5)} / {len(ranks)}")
    print(f"The Top-10 recovery rate is: {sum(ranks < 10) / len(ranks) :.3f}, {sum(ranks < 10)} / {len(ranks)}")
    print(f"The Top-15 recovery rate is: {sum(ranks < 15) / len(ranks) :.3f}, {sum(ranks < 15)} / {len(ranks)}")
    print(f"The Top-30 recovery rate is: {sum(ranks < 30) / len(ranks) :.3f}, {sum(ranks < 30)} / {len(ranks)}")
    print()

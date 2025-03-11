import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.realpath(".."))
sys.path.append('/home/super/Desktop/AmortizedCausalDiscovery/')
sys.path.append('/home/super/Desktop/AmortizedCausalDiscovery/codebase')

import torch

from model.modules import *
from model.Encoder import Encoder


class MLPEncoder(Encoder):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(self, args, n_in, n_hid, n_out, do_prob=0.0, factor=True):
        super().__init__(args, factor)

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")

        if args.discrete:
            self.fc_out = nn.Linear(n_hid, n_out)
        else:
            self.fc_out1 = nn.Linear(n_hid, 1)
            self.fc_out2 = nn.Linear(n_hid, 1)

        # self.batchnorm_mean = nn.BatchNorm1d(n_out, affine=False)
        # self.scaler = Scaler()
        # self.batchnorm_std = nn.BatchNorm1d(n_out, affine=False)

        self.init_weights()

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        x = self.mlp1(x)  # 2-layer ELU net per node
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        if self.args.discrete:
            return self.fc_out(x)
        else:
            z_mean = self.fc_out1(x).squeeze(-1)
            z_log_var = self.fc_out2(x).squeeze(-1)

            # z_mean = self.scaler(self.batchnorm_mean(z_mean), mode= 'positive')
            # z_log_var = self.scaler(self.batchnorm_std(z_log_var), mode='negative')

            return z_mean, z_log_var


if __name__ == "__main__":
    from codebase.utils import arg_parser
    args = arg_parser.parse_args()
    device = "cuda:6" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    model = MLPEncoder(
                args,
                args.timesteps * args.dims,
                args.encoder_hidden,
                args.edge_types,
                do_prob=args.encoder_dropout,
                factor=args.factor,
            ).to(device)
    sample = torch.randn(128, 3, 12, 1).to(device)
    rel_rec, rel_send = utils.create_rel_rec_send(args, args.num_atoms)
    print(rel_rec, rel_send)
    result1 = model(sample, rel_rec, rel_send)
    print(result1.shape)
    # edges = utils.gumbel_softmax(result, tau=args.temp, hard=args.hard)
    # print(edges.shape)
    
    # import numpy as np
    # def encode_onehot(labels):
    #     """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    #     classes = set(labels)
    #     classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    #     labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    #     return labels_onehot
    
    # off_diag = np.ones([3, 3]) - np.eye(3)
    # # print(off_diag)
    # # print(np.where(off_diag))

    # rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    # rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    # print(rel_send)
    # print(rel_rec)

    # a = np.arange(9)
    # print(a)
    # print(a.reshape(3, 3))




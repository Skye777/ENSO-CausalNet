import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.realpath(".."))
sys.path.append('/home/super/Desktop/AmortizedCausalDiscovery/')
sys.path.append('/home/super/Desktop/AmortizedCausalDiscovery/codebase')

import torch

from model.modules import *
from model import utils


class RNNDecoder(nn.Module):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(self, args, n_in_node, edge_types, n_hid, do_prob=0.0, skip_first=False):
        super(RNNDecoder, self).__init__()
        self.args = args

        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_i = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_n = nn.Linear(n_in_node, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print("Using learned recurrent interaction net decoder.")

        self.dropout_prob = do_prob

    def single_step_forward(self, inputs, rel_rec, rel_send, rel_type, hidden):

        # node2edge
        receivers = torch.matmul(rel_rec, hidden)
        senders = torch.matmul(rel_send, hidden)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape)

        if inputs.is_cuda:
            all_msgs = all_msgs.to(self.args.device)

        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.msg_fc2)) - 1.0
        else:
            start_idx = 0
            norm = float(len(self.msg_fc2))
        # print("print", start_idx, len(self.msg_fc2))
        # Run separate MLP for every edge type
        # NOTE: To exclude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            # print("Hey! Here is considered")
            msg = torch.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = torch.tanh(self.msg_fc2[i](msg))
            msg = msg * rel_type[:, :, i : i + 1]
            all_msgs += msg / norm

        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous() / inputs.size(2)  # Average

        # GRU-style gated aggregation
        r = torch.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
        i = torch.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
        n = torch.tanh(self.input_n(inputs) + r * self.hidden_h(agg_msgs))
        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = F.dropout(F.leaky_relu(self.out_fc1(hidden)), p=self.dropout_prob)
        pred = F.dropout(F.leaky_relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        if self.args.auto == False:
            # Predict position/velocity difference
            pred = inputs + pred

        return pred, hidden

    def forward(
        self,
        data,
        rel_type,
        rel_rec,
        rel_send,
        pred_steps=1,
        burn_in=False,
        burn_in_steps=1,
        dynamic_graph=False,
        encoder=None,
        temp=None,
    ):
        if dynamic_graph:
            inputs = data[:, :, self.args.timesteps-burn_in_steps-1: , :].transpose(1, 2).contiguous()
        else:
            inputs = data.transpose(1, 2).contiguous()

        time_steps = inputs.size(1)

        if not self.args.discrete:
            rel_type = rel_type.unsqueeze(-1)

        # inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]

        hidden = torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape)

        if inputs.is_cuda:
            hidden = hidden.to(self.args.device)

        pred_all = []
        causal_graph = []

        for step in range(0, inputs.size(1) - 1):

            if burn_in:
                if step <= burn_in_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]
            else:
                assert pred_steps <= time_steps
                # Use ground truth trajectory input vs. last prediction
                if not step % pred_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]

            if dynamic_graph and step > burn_in_steps:
                # print("Here!")
                # NOTE: Assumes burn_in_steps = args.timesteps
                # logits = encoder(
                #     data[:, :, step - burn_in_steps : step, :].contiguous(),
                #     rel_rec,
                #     rel_send,
                # )
                if step - burn_in_steps < self.args.timesteps:
                    data_in = torch.cat((data[:, :, step - burn_in_steps : self.args.timesteps, :], torch.stack(pred_all[burn_in_steps:], dim=2)), dim=2)
                else:
                    data_in = torch.stack(pred_all[-self.args.timesteps:], dim=2)
                # print(data[:, :, step - burn_in_steps : self.args.timesteps, :].shape, torch.stack(pred_all[burn_in_steps:], dim=2).shape)
                logits = encoder(
                    # data[:, :, step - burn_in_steps : step + self.args.timesteps - burn_in_steps, :].contiguous(),
                    data_in.contiguous(),
                    rel_rec,
                    rel_send,
                )
                rel_type = utils.gumbel_softmax(logits, tau=temp, hard=True)
                causal_graph.append(logits)

            pred, hidden = self.single_step_forward(
                ins, rel_rec, rel_send, rel_type, hidden
            )
            # print(pred.shape)
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)
        if len(causal_graph) > 0:
            causal_graph = torch.stack(causal_graph, dim=1)

        return preds.transpose(1, 2).contiguous(), causal_graph


if __name__ == "__main__":
    from codebase.utils import arg_parser
    args = arg_parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # print(x[0::10])
    model = RNNDecoder(
                args=args,
                n_in_node=args.dims,
                edge_types=args.edge_types,
                n_hid=args.decoder_hidden,
                do_prob=args.decoder_dropout,
                skip_first=args.skip_first,
            ).to(device)
    sample = torch.randn(128, 8, 12, 1).to(device)
    edges = torch.randn(128, 56, args.edge_types).to(device)
    rel_rec, rel_send = utils.create_rel_rec_send(args, args.num_atoms)
    print(rel_rec.shape, rel_send.shape)
    result, causals = model(
            sample,
            edges,
            rel_rec,
            rel_send,
            pred_steps=args.prediction_steps,
            burn_in=True,
            burn_in_steps=args.burn_in,
        )
    print(result.shape)
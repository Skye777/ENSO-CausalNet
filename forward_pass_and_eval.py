from __future__ import division
from __future__ import print_function

from collections import defaultdict
import time
import torch

import numpy as np

from model.modules import *
from model import utils, utils_unobserved


def test_time_adapt(
    args,
    logits,
    decoder,
    data_encoder,
    rel_rec,
    rel_send,
    predicted_atoms,
    log_prior,
):
    with torch.enable_grad():
        tta_data_decoder = data_encoder.detach()

        if args.use_encoder:
            ### initialize q(z) with q(z|x)
            tta_logits = logits.detach()
            tta_logits.requires_grad = True
        else:
            ### initialize q(z) randomly
            tta_logits = torch.randn_like(
                logits, device=args.device.type, requires_grad=True
            )

        tta_optimizer = torch.optim.Adam(
            [{"params": tta_logits, "lr": args.lr_logits}]
        )
        tta_target = data_encoder[:, :, 1:, :].detach()

        ploss = 0
        for i in range(args.num_tta_steps):
            tta_optimizer.zero_grad()

            tta_edges = utils.gumbel_softmax(tta_logits, tau=args.temp, hard=False)

            tta_output = decoder(
                tta_data_decoder, tta_edges, rel_rec, rel_send, args.prediction_steps
            )

            loss = utils.nll_gaussian(tta_output, tta_target, args.var)

            prob = utils.my_softmax(tta_logits, -1)

            if args.prior != 1:
                loss += utils.kl_categorical(prob, log_prior, predicted_atoms) 
            else:
                loss += utils.kl_categorical_uniform(
                    prob, predicted_atoms, args.edge_types
                ) 

            loss.backward()
            tta_optimizer.step()
            ploss += loss.cpu().detach()

            if i == 0:
                first_loss = loss.cpu().detach()
            if (i + 1) % 10 == 0:
                print(i, ": ", ploss / 10)
                ploss = 0

    print("Fine-tuning improvement: ", first_loss - loss.cpu().detach())

    return tta_logits


def forward_pass_and_eval(
    args,
    encoder,
    decoder,
    data,
    rel_rec,
    rel_send,
    hard,
    data_encoder=None,
    data_decoder=None,
    edge_probs=None,
    val=False,
    log_prior=None,
    return_outputs=False,
    dynamic_graph=False
):
    start = time.time()
    losses = defaultdict(lambda: torch.zeros((), device=args.device))

    #################### INPUT DATA ####################
    if data_encoder is None:
        data_encoder = data
    if data_decoder is None:
        data_decoder = data

    predicted_atoms = args.num_atoms
    #################### ENCODER ####################
    if args.use_encoder:
        ## model only the edges
        if args.discrete:
            logits = encoder(data_encoder, rel_rec, rel_send)   # (B, N*N-N, edge_type)
            edges = utils.gumbel_softmax(logits, tau=args.temp, hard=hard)   # (B, N*N-N, edge_type)
            prob = utils.my_softmax(logits, -1)   # (B, N*N-N, edge_type)
        else:
            z_mean, z_log_var = encoder(data_encoder, rel_rec, rel_send)   # (B, N*N-N)
            edges = utils.sampling(z_mean, z_log_var)
    else:
        logits = edge_probs.unsqueeze(0).repeat(data_encoder.shape[0], 1, 1)

    # result = {'logits': logits.cpu().detach().numpy(), "edges": edges.cpu().detach().numpy(), 'prob': prob.cpu().detach().numpy()}
    # np.savez("z15.npz", **result)

    target = data_decoder[:, :, 1:, :]

    #################### DECODER ####################
    if args.decoder == "rnn":
        output, update_logits = decoder(
            data_decoder,
            edges,
            rel_rec,
            rel_send,
            pred_steps=args.prediction_steps,
            burn_in=True,
            # burn_in_steps=args.timesteps - args.prediction_steps,
            burn_in_steps=args.burn_in,
            dynamic_graph=dynamic_graph,
            encoder=encoder, 
            temp=args.temp,
        )
    else:
        output = decoder(
            data_decoder,
            edges,
            rel_rec,
            rel_send,
            args.prediction_steps,
        )
    
    # Testing
    if return_outputs:
        # print("logits:", update_logits.shape)
        all_logits = torch.cat((logits.unsqueeze(1), update_logits), dim=1)
        return output, all_logits
    if val:
        # print(output.shape, target.shape)
        output = output[:, 0, args.burn_in:, 0]
        target = target[:, 0, args.timesteps-1:, 0]
        # print(output.shape, target.shape)
        losses['cor'] = utils.calscore(output, target)
    #################### MAIN LOSSES ####################
    ### latent losses ###
    if args.discrete:
        losses["loss_kl"] = utils.kl_latent(args, prob, log_prior, predicted_atoms)
    else:
        losses["loss_kl"] = utils.kl_loss(z_mean, z_log_var)
    # losses["acc"] = utils.edge_accuracy(logits, relations)
    # losses["auroc"] = utils.calc_auroc(prob, relations)
    # print(losses["loss_kl"])

    ### output losses ###
    losses["loss_nll"] = utils.nll_gaussian(
        output, target, args.var
    ) 
    ### nino losses ###  
    # losses["loss_nino"] = utils.nino_loss(output[:, 0, :, 0], target[:, 0, :, 0])

    losses["loss_mse"] = F.mse_loss(output, target)

    total_loss = losses["loss_nll"] + losses["loss_kl"]
    # print(losses["mse_unobserved"].device, total_loss.device)
    # total_loss += args.teacher_forcing * losses["mse_unobserved"]
    if args.global_temp:
        total_loss += losses['loss_kl_temp']
    losses["loss"] = total_loss

    losses["inference time"] = time.time() - start

    return losses, output




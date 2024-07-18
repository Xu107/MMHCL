import os
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from utility.parser import parse_args
from utility.norm import build_sim, build_knn_normalized_graph
args = parse_args()

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim


        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

    def forward(self, adj):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(args.UI_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings



class MMHCL(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super(MMHCL, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embeddings_dim = embedding_dim

        self.user_ui_embedding = nn.Embedding(n_users, self.embeddings_dim)
        self.item_ui_embedding = nn.Embedding(n_items, self.embeddings_dim)

        self.uu_embedding = nn.Embedding(n_users, self.embeddings_dim)
        self.ii_embedding = nn.Embedding(n_items, self.embeddings_dim)

        if args.cf_model == 'NGCF':
            self.GC_Linear_list = nn.ModuleList()
            self.Bi_Linear_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            for i in range(args.UI_layers):
                self.GC_Linear_list.append(nn.Linear(eval(args.weight_size)[i], eval(args.weight_size)[i + 1]))
                self.Bi_Linear_list.append(nn.Linear(eval(args.weight_size)[i], eval(args.weight_size)[i + 1]))
                self.dropout_list.append(nn.Dropout(0.1))

        nn.init.xavier_uniform_(self.user_ui_embedding.weight)
        nn.init.xavier_uniform_(self.item_ui_embedding.weight)
        nn.init.xavier_uniform_(self.uu_embedding.weight)
        nn.init.xavier_uniform_(self.ii_embedding.weight)

        self.tau = args.temperature

    def forward(self, UI_mat, I2I_mat, U2U_mat):

        ii_emb = self.ii_embedding.weight
        uu_emb = self.uu_embedding.weight

        if args.item_loss_ratio != 0:
            for i in range(args.Item_layers):
                ii_emb = torch.sparse.mm(I2I_mat, ii_emb)

        if args.user_loss_ratio != 0:
            for i in range(args.User_layers):
                uu_emb = torch.sparse.mm(U2U_mat, uu_emb)

        if args.cf_model == 'LightGCN':
            ego_embeddings = torch.cat((self.user_ui_embedding.weight, self.item_ui_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(args.UI_layers):
                side_embeddings = torch.sparse.mm(UI_mat, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_ui_emb, i_ui_emb = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        elif args.cf_model == 'NGCF':
            ego_embeddings = torch.cat((self.user_ui_embedding.weight, self.item_ui_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(args.UI_layers):
                side_embeddings = torch.sparse.mm(UI_mat, ego_embeddings)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)

                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]

            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_ui_emb, i_ui_emb = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        elif args.cf_model == 'MF':
            u_ui_emb, i_ui_emb=self.user_ui_embedding.weight, self.item_ui_embedding.weight

        if args.item_loss_ratio != 0:
            i_ui_emb = i_ui_emb + F.normalize(ii_emb, p=2, dim=1)

        if args.user_loss_ratio != 0:
            u_ui_emb = u_ui_emb + F.normalize(uu_emb, p=2, dim=1)



        return u_ui_emb, i_ui_emb, ii_emb, uu_emb

    def batched_contrastive_loss(self, z1, z2, batch_size=4096):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())


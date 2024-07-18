import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import json
from utility.parser import parse_args

args = parse_args()

import torch



class Data(object):
    def __init__(self, path, batch_size):
        self.path = path + '/%d-core' % args.core
        self.batch_size = batch_size

        train_file = path + '/%d-core/train.json' % (args.core)
        val_file = path + '/%d-core/val.json' % (args.core)
        test_file = path + '/%d-core/test.json' % (args.core)

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test, self.n_val = 0, 0, 0
        self.neg_pools = {}

        self.exist_users = []

        train = json.load(open(train_file))
        test = json.load(open(test_file))
        val = json.load(open(val_file))
        for uid, items in train.items():
            if len(items) == 0:
                continue
            uid = int(uid)
            self.exist_users.append(uid)
            self.n_items = max(self.n_items, max(items))
            self.n_users = max(self.n_users, uid)
            self.n_train += len(items)

        for uid, items in test.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_test += len(items)
            except:
                continue

        for uid, items in val.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_val += len(items)
            except:
                continue

        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_Item_Interacts = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)

        self.train_items, self.test_set, self.val_set = {}, {}, {}
        for uid, train_items in train.items():
            if len(train_items) == 0:
                continue
            uid = int(uid)
            for idx, i in enumerate(train_items):
                self.R[uid, i] = 1.

            self.train_items[uid] = train_items

        for uid, test_items in test.items():
            uid = int(uid)
            if len(test_items) == 0:
                continue
            try:
                self.test_set[uid] = test_items
            except:
                continue

        for uid, val_items in val.items():
            uid = int(uid)
            if len(val_items) == 0:
                continue
            try:
                self.val_set[uid] = val_items
            except:
                continue

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        # users = self.exist_users[:]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u(u, 3)
        return users, pos_items, neg_items

    # 原始的Latiice
    # general Model 返回的是numpy类型的，度归一化用-1
    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def get_D_inv(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            return d_mat_inv

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    # ---------------------------------------Own--------------------------------------------------
    def norm_dense(self, adj, normalization='origin'):
        if normalization == 'sym':
            rowsum = torch.sum(adj, -1)
            d_inv_sqrt = torch.pow(rowsum, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
            L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        elif normalization == "2sym":
            rowsum = torch.sum(adj, -1)
            d_row_inv_sqrt = torch.pow(rowsum, -0.5)
            d_row_inv_sqrt[torch.isinf(d_row_inv_sqrt)] = 0.
            d_row_mat_inv_sqrt = torch.diagflat(d_row_inv_sqrt)

            colsum = torch.sum(adj, -2)
            d_col_inv_sqrt = torch.pow(colsum, -0.5)
            d_col_inv_sqrt[torch.isinf(d_col_inv_sqrt)] = 0.
            d_col_mat_inv_sqrt = torch.diagflat(d_col_inv_sqrt)

            L_norm = torch.mm(torch.mm(d_row_mat_inv_sqrt, adj), d_col_mat_inv_sqrt)

        elif normalization == 'rw':
            rowsum = torch.sum(adj, -1)
            d_inv = torch.pow(rowsum, -1)
            d_inv[torch.isinf(d_inv)] = 0.
            d_mat_inv = torch.diagflat(d_inv)
            L_norm = torch.mm(d_mat_inv, adj)
        elif normalization == 'origin':
            L_norm = adj
        return L_norm

    def get_UI_mat(self, norm_type='sym'):
        #   UI_mat default use sym normalization,and No-self-connection
        print("Loading UI_mat:(" + norm_type + ")")
        t = time()
        try:
            UI_mat = torch.load(self.path + '/UI_mat_' + norm_type + ".pth")
        except Exception:
            adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.R.tolil()

            adj_mat[:self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, :self.n_users] = R.T
            adj_mat = adj_mat.todense()
            UI_mat = torch.from_numpy(adj_mat).float()
            UI_mat = self.norm_dense(UI_mat, norm_type)
            UI_mat = UI_mat.to_sparse()
            torch.save(UI_mat, self.path + '/UI_mat_' + norm_type + ".pth")
        print("End Load UI_mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return UI_mat

    def get_UI_single_mat(self, norm_type='2sym'):
        print("Loading UI_single_mat:(" + norm_type + ")")
        t = time()
        try:
            UI_mat = torch.load(self.path + '/UI_single_mat_' + norm_type + ".pth")
        except Exception:
            adj_mat = self.R.todense()
            UI_mat = torch.from_numpy(adj_mat).float()
            UI_mat = self.norm_dense(UI_mat, norm_type)
            UI_mat = UI_mat.to_sparse()
            torch.save(UI_mat, self.path + '/UI_single_mat_' + norm_type + ".pth")
        print("End Load UI_single_mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return UI_mat

    def get_U2U_mat(self, norm_type='rw'):
        # U2U_mat default use row normalization,and No-self-connection
        print("Loading User_mat:(" + norm_type + ")")
        t = time()
        try:
            User_mat = torch.load(self.path + '/User_mat_' + norm_type + ".pth")
        except Exception:
            R = torch.from_numpy(self.R.todense()).float()
            User_mat = R @ R.T
            n_user = User_mat.size()[0]
            mask = torch.eye(n_user)
            User_mat[mask > 0] = 0  # 抹去自连接
            User_mat = self.norm_dense(User_mat, norm_type)
            User_mat = User_mat.to_sparse()
            torch.save(User_mat, self.path + '/User_mat_' + norm_type + ".pth")
        print("End Load User_mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return User_mat

    def get_I2I_single_mat(self, norm_type="sym"):
        # I2I_mat default use sym normalization,and must have self-connection because of similarity
        print("Loading I2I media-specific mat:(" + norm_type + ")")
        t = time()
        try:
            image_adj = torch.load(self.path + '/Image_mat_' + norm_type + ".pth")
            text_adj = torch.load(self.path + '/Text_mat_' + norm_type + ".pth")
            if args.dataset=="tiktok":
                audio_adj = torch.load(self.path + '/Audio_mat_' + norm_type + ".pth")

        except Exception:
            # image_feats = np.load('../data/old/{}/image_feat.npy'.format(args.dataset))  # '../data/{}/image_feat.npy'
            image_feats = np.load('../data/{}/image_feat.npy'.format(args.dataset))  # '../data/{}/image_feat.npy'
            # text_feats = np.load('../data/old/{}/text_feat.npy'.format(args.dataset))
            text_feats = np.load('../data/{}/text_feat.npy'.format(args.dataset))
            if args.dataset == "tiktok":
                # audio_feats = np.load('../data/old/{}/audio_feat.npy'.format(args.dataset))
                audio_feats = np.load('../data/{}/audio_feat.npy'.format(args.dataset))

            image_feats = torch.tensor(image_feats).float()
            text_feats = torch.tensor(text_feats).float()
            if args.dataset == "tiktok":
                audio_feats = torch.tensor(audio_feats).float()

            image_adj = self.build_sim(image_feats)
            image_adj = self.build_knn_normalized_graph(image_adj, topk=args.topk)

            text_adj = self.build_sim(text_feats)
            text_adj = self.build_knn_normalized_graph(text_adj, topk=args.topk)
            if args.dataset == "tiktok":
                audio_adj = self.build_sim(audio_feats)
                audio_adj = self.build_knn_normalized_graph(audio_adj, topk=args.topk)

            image_adj = self.norm_dense(image_adj, norm_type)
            text_adj = self.norm_dense(text_adj, norm_type)
            if args.dataset == "tiktok":
                audio_adj = self.norm_dense(audio_adj, norm_type)


            image_adj = image_adj.to_sparse()
            text_adj = text_adj.to_sparse()
            if args.dataset == "tiktok":
                audio_adj = audio_adj.to_sparse()


            torch.save(image_adj, self.path + '/Image_mat_' + norm_type + ".pth")
            torch.save(text_adj, self.path + '/Text_mat_' + norm_type + ".pth")
            if args.dataset == "tiktok":
                torch.save(audio_adj, self.path + '/Audio_mat_' + norm_type + ".pth")

        print("End Load I2I media-specific mat:[%.1fs](" % (time() - t) + norm_type + ")")
        if args.dataset == "tiktok":
            return image_adj, text_adj, audio_adj
        else:
            return image_adj, text_adj, ""

    # Order to speed up when Model forward this is be replaced
    def get_I2I_Hypergrah_mat(self, norm_type="origin"):
        # I2I_Hypergraph_mat use origin normalization
        print(f"Loading I2I multi-media Hypergraph mat:({norm_type})_topk:{str(args.topk)}")
        t = time()
        try:
            Hypergraph = torch.load(f"{self.path}/hypergraph_mat_{norm_type}_topk_{str(args.topk)}.pth")
        except Exception:
            # image_feats = np.load('../data/old/{}/image_feat.npy'.format(args.dataset))  # '../data/{}/image_feat.npy'
            # text_feats = np.load('../data/old/{}/text_feat.npy'.format(args.dataset))
            image_feats = np.load('../data/{}/image_feat.npy'.format(args.dataset))  # '../data/{}/image_feat.npy'
            text_feats = np.load('../data/{}/text_feat.npy'.format(args.dataset))

            image_feats = torch.tensor(image_feats).float()
            text_feats = torch.tensor(text_feats).float()

            image_adj = self.build_sim(image_feats)
            image_adj = self.build_knn_normalized_graph(image_adj, topk=args.topk)

            text_adj = self.build_sim(text_feats)
            text_adj = self.build_knn_normalized_graph(text_adj, topk=args.topk)

            Hypergraph = torch.cat((image_adj, text_adj), dim=1)
            Hypergraph = self.norm_dense(Hypergraph, norm_type)
            Hypergraph = Hypergraph.to_sparse()
            torch.save(Hypergraph, f"{self.path}/hypergraph_mat_{norm_type}_topk_{str(args.topk)}.pth")
        print("End Load I2I multi-media Hypergraph mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph

    def get_I2I_Hypergraph_mul_mat(self, norm_type="sym"):
        # I2I_Hypergraph_mat*I2I_Hypergraph_mat.T use sys normalization
        print(f"Loading I2I multi-media Hypergraph mul mat*mat.T:({norm_type})_topk:{str(args.topk)}")
        t = time()
        try:
            Hypergraph_mul = torch.load(f"{self.path}/hypergraph_mat_mul_{norm_type}_topk_{str(args.topk)}.pth")
        except Exception:
            Hypergraph = self.get_I2I_Hypergrah_mat("origin")
            Hypergraph_mul = torch.sparse.mm(Hypergraph, Hypergraph.to_dense().T)
            Hypergraph_mul = self.norm_dense(Hypergraph_mul, norm_type)
            Hypergraph_mul = Hypergraph_mul.to_sparse()
            torch.save(Hypergraph_mul, f"{self.path}/hypergraph_mat_mul_{norm_type}_topk_{str(args.topk)}.pth")
        print("End Load I2I multi-media Hypergraph mul mat*mat.T:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph_mul

    #pytorch---------------------------------------------------------------------------------------------------------------
    def get_I2I_Hypergrah_mat_pt(self, norm_type="origin"):
        # I2I_Hypergraph_mat use origin normalization
        print("Loading I2I multi-media Hypergraph mat:(" + norm_type + ")")
        t = time()
        try:
            Hypergraph = torch.load(self.path + '/hypergraph_mat_' + norm_type + ".pth")
        except Exception:
            image_feats = torch.load("../data/{}/img_feat.pt".format(args.dataset))
            text_feats=torch.load("../data/{}/text_feat.pt".format(args.dataset))

            # image_feats = torch.tensor(image_feats).float()
            # text_feats = torch.tensor(text_feats).float()

            # image_adj = self.build_sim_feature_nan(image_feats)
            image_adj = self.build_sim(image_feats)
            image_adj = self.build_knn_normalized_graph(image_adj, topk=args.topk)

            # text_adj = self.build_sim_feature_nan(text_feats)
            text_adj = self.build_sim(text_feats)
            text_adj = self.build_knn_normalized_graph(text_adj, topk=args.topk)

            Hypergraph = torch.cat((image_adj, text_adj), dim=1)
            Hypergraph = self.norm_dense(Hypergraph, norm_type)
            Hypergraph = Hypergraph.to_sparse()
            torch.save(Hypergraph, self.path + '/hypergraph_mat_' + norm_type + ".pth")
        print("End Load I2I multi-media Hypergraph mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph
    #pytorch
    def get_I2I_Hypergraph_mul_mat_pt(self,norm_type="sym"):
        # I2I_Hypergraph_mat*I2I_Hypergraph_mat.T use sys normalization
        print("Loading I2I multi-media Hypergraph mul mat*mat.T pytorch:(" + norm_type + ")")
        t = time()
        try:
            Hypergraph_mul = torch.load(self.path + '/hypergraph_mat_mul' + norm_type + ".pth")
        except Exception:
            Hypergraph = self.get_I2I_Hypergrah_mat_pt("origin")
            Hypergraph_mul = torch.sparse.mm(Hypergraph, Hypergraph.to_dense().T)
            Hypergraph_mul = self.norm_dense(Hypergraph_mul, norm_type)
            Hypergraph_mul = Hypergraph_mul.to_sparse()
            torch.save(Hypergraph_mul, self.path + '/hypergraph_mat_mul' + norm_type + ".pth")
        print("End Load I2I multi-media Hypergraph mul mat*mat.T pytorch:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph_mul
    #---------------------------------------------------------------------------------------------------------------------

    # Order to speed up when Model forward this is be replaced
    def get_tiktok_I2I_Hypergrah_mat(self, norm_type="origin"):
        # I2I_Hypergraph_mat use origin normalization
        print(f"Loading I2I multi-media Hypergraph mat:({ norm_type })_topk:{str(args.topk)}")
        t = time()
        try:
            Hypergraph = torch.load(f"{self.path}/hypergraph_mat_{norm_type}_topk_{str(args.topk)}.pth")
        except Exception:
            # image_feats = np.load('../data/old/{}/image_feat.npy'.format(args.dataset))  # '../data/{}/image_feat.npy'
            # text_feats = np.load('../data/old/{}/text_feat.npy'.format(args.dataset))
            # audio_feats = np.load('../data/old/{}/audio_feat.npy'.format(args.dataset))
            image_feats = np.load('../data/{}/image_feat.npy'.format(args.dataset))  # '../data/{}/image_feat.npy'
            text_feats = np.load('../data/{}/text_feat.npy'.format(args.dataset))
            audio_feats = np.load('../data/{}/audio_feat.npy'.format(args.dataset))

            image_feats = torch.tensor(image_feats).float()
            text_feats = torch.tensor(text_feats).float()
            audio_feats = torch.tensor(audio_feats).float()

            image_adj = self.build_sim(image_feats)
            image_adj = self.build_knn_normalized_graph(image_adj, topk=args.topk)

            text_adj = self.build_sim(text_feats)
            text_adj = self.build_knn_normalized_graph(text_adj, topk=args.topk)

            audio_adj = self.build_sim(audio_feats)
            audio_adj = self.build_knn_normalized_graph(audio_adj, topk=args.topk)

            Hypergraph = torch.cat((torch.cat((image_adj, text_adj), dim=1), audio_adj), dim=1)
            Hypergraph = self.norm_dense(Hypergraph, norm_type)
            Hypergraph = Hypergraph.to_sparse()
            torch.save(Hypergraph, f"{self.path}/hypergraph_mat_{norm_type}_topk_{str(args.topk)}.pth")
        print("End Load I2I multi-media Hypergraph mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph

    def get_tiktok_I2I_Hypergraph_mul_mat(self, norm_type="sym"):
        # I2I_Hypergraph_mat*I2I_Hypergraph_mat.T use sys normalization
        print(f"Loading I2I multi-media Hypergraph mul mat*mat.T:({norm_type})_topk:{str(args.topk)}")
        t = time()
        try:
            Hypergraph_mul = torch.load(f"{self.path}/hypergraph_mat_mul_{norm_type}_topk_{str(args.topk)}.pth")
        except Exception:
            Hypergraph = self.get_tiktok_I2I_Hypergrah_mat("origin")
            Hypergraph_mul = torch.sparse.mm(Hypergraph, Hypergraph.to_dense().T)
            Hypergraph_mul = self.norm_dense(Hypergraph_mul, norm_type)
            Hypergraph_mul = Hypergraph_mul.to_sparse()
            torch.save(Hypergraph_mul, f"{self.path}/hypergraph_mat_mul_{norm_type}_topk_{str(args.topk)}.pth")
        print("End Load I2I multi-media Hypergraph mul mat*mat.T:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph_mul


    def build_sim(self, context):
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        return sim

    def build_sim_feature_nan(self, context):
        #image feature extract when url is unvalid or image is destroy features=0,if use norm will nan
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
        context_norm[context_norm.isnan()] = 0
        sim = torch.mm(context, context.transpose(1, 0))
        return sim

    def build_knn_normalized_graph(self, adj, topk):
        knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
        adj = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        adj[adj > 0] = 1.
        return adj

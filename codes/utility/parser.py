import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--data_path', nargs='?', default='../data/',
                        help='Input data path.')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--dataset', nargs='?', default='Tiktok',
                        help='Choose a dataset from {Tiktok,Sports,Clothing}')
    parser.add_argument('--verbose', type=int, default=5,#5
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1000,#1000,
                        help='Number of epoch.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--regs', type=float, default=1e-3,
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--train_dir', default='train')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--weight_size', nargs='?', default='[64,64,64]',
                        help='Output sizes of every layer')
    parser.add_argument('--core', type=int, default=5,
                        help='5-core for warm-start; 0-core for cold start')
    parser.add_argument('--topk', type=int, default=5,
                        help='K value of k-NN sparsification')
    parser.add_argument('--cf_model', nargs='?', default='LightGCN',
                        help='Downstream Collaborative Filtering model {MF, NGCF, LightGCN}')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='')

    parser.add_argument('--sparse', type=int, default=0, help='Sparse or dense adjacency matrix')
    parser.add_argument('--debug', default="True")

    parser.add_argument('--norm_type', nargs='?', default='sym', help='Adjacency matrix normalization operation')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id')

    parser.add_argument('--Ks', nargs='?', default='[10,20]',
                        help='K value of ndcg/recall @ k')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')


    parser.add_argument('--UI_layers', type=int, default=2,
                        help='UI GNN layers')
    parser.add_argument('--User_layers', type=int, default=3,
                        help='UI GNN layers')
    parser.add_argument('--Item_layers', type=int, default=2,
                        help='UI GNN layers')
    parser.add_argument('--user_loss_ratio', type=float, default=0.03,
                        help='Control the effect of the contrastive auxiliary task')
    parser.add_argument('--item_loss_ratio', type=float, default=0.07,
                        help='Control the effect of the contrastive auxiliary task')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='InfoNCE temperature')

    parser.add_argument('--ablation_target', type=str, default="",
                        help='UI GNN layers')

    return parser.parse_args()

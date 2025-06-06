# run collaborative filtering (variants) algorithm

import os
import argparse
import pickle
from utils import bool_ext

__author__ = "Jimmy Shong"


parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="manual seed", type=int, default=2019)

parser.add_argument("--is_train", help="whether to train or not (tune)", type=bool_ext, default=True)
parser.add_argument("--eval", help="whether to only evaluate from a saved pkl file", type=bool_ext, default=False)
parser.add_argument("--pkl_path", help="path to the .pkl file for evaluation", type=str, default=None)
parser.add_argument("--store_model", help="whether to store model pkl file", type=bool_ext, default=False)

parser.add_argument("--input_dir", help="directory of input files", type=str, default="data/input")
parser.add_argument("--output_dir", help="directory of output files", type=str, default="data/output/cf/")
parser.add_argument("--repository", help="data to be analyzed, can be gdsc or ccle", type=str, default="gdsc")#gdsc
parser.add_argument("--drug_id", help="the index of drug to be predicted in STL, -1 if MTL", type=int, default=-1)#0-259
parser.add_argument("--use_cuda", help="whether to use GPU or not", type=bool_ext, default=True)
# parser.add_argument("--use_relu", help="whether to use relu or not", type=bool_ext, default=False)
parser.add_argument("--use_relu", help="whether to use relu or not", type=bool_ext, default=True)
parser.add_argument("--init_gene_emb", help="whether to use pretrained gene embedding or not", type=bool_ext, default=True)
# parser.add_argument("--use_oc", help="whether to use onecycle or not", type=bool_ext, default=True)
parser.add_argument("--scheduler", help="determine what schedule to use (onecycle, cosine, or none)", type=str, default="onecycle")
parser.add_argument("--shuffle", help="whether to shuffle", type=bool_ext, default=False)

parser.add_argument("--omic", help="type of omics data, can be exp, mut, cnv, met, mul", type=str, default="exp")

parser.add_argument("--use_attention", help="whether to use attention mechanism or not", type=bool_ext, default=True)
parser.add_argument("--use_cntx_attn", help="whether to use contextual attention or not", type=bool_ext, default=True)

parser.add_argument("--embedding_dim", help="embedding dimension", type=int, default=200) #200
parser.add_argument("--attention_size", help="size of attention parameter beta_j", type=int, default=128) #150
parser.add_argument("--attention_head", help="number of attention heads", type=int, default=8) #8
parser.add_argument("--hidden_dim_enc", help="dimension of hidden layer in encoder", type=int, default=200) #200
# parser.add_argument("--use_hid_lyr", help="whether to use hidden layer in the encoder or not", type=bool_ext, default=False)
parser.add_argument("--use_hid_lyr", help="whether to use hidden layer in the encoder or not", type=bool_ext, default=True)

# parser.add_argument("--max_iter", help="maximum number of training iterations", type=int, default=int(384000))
parser.add_argument("--max_iter", help="maximum number of training iterations", type=int, default=int(48000))
# parser.add_argument("--max_fscore", help="maximum f1 score", type=float, default=0.6)
# parser.add_argument("--max_fscore", help="maximum f1 score", type=float, default=0.6452)
parser.add_argument("--max_fscore", help="maximum f1 score", type=float, default=-1)
parser.add_argument("--dropout_rate", help="probability of an element to be zero-ed", type=float, default=0.6)#0.3

parser.add_argument("--learning_rate", help="learning rate for optimizer", type=float, default=0.3)
parser.add_argument("--weight_decay", help="coefficient of l2 regularizer", type=float, default=3e-4)#3e-4
parser.add_argument("--batch_size", help="training batch size", type=int, default=8)#256
parser.add_argument("--test_batch_size", help="test batch size", type=int, default=8)
parser.add_argument("--test_inc_size", help="increment interval size between log outputs", type=int, default=1024)#64

# parser.add_argument("--model_label", help="model name", type=str, default="CF")
parser.add_argument("--model_label", help="model name", type=str, default="cntx-attn-gdsc")

parser.add_argument("--focal", help="whether to use focal or not", type=bool_ext, default=False)
parser.add_argument("--alpha", help="alpha for focal loss", type=float, default=0.6)
parser.add_argument("--gamma", help="gamma for focal loss", type=float, default=2.0)

parser.add_argument("--adam", help="whether to use AdamW or not", type=bool_ext, default=False)

parser.add_argument("--mlp", help="whether to use MLP in decoder or not", type=bool_ext, default=False)

parser.add_argument("--norm_strategy", help="whether to use postnorm, prenorm, or none", type=str, default="None")
parser.add_argument("--use_residual", help="whether to use residuals", type=bool_ext, default=False)

args = parser.parse_args()

import random
import numpy as np

SEED = args.seed
print("SEED:", SEED)
random.seed(SEED)
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

from utils import fill_mask, load_dataset, split_dataset

from collabfilter import CF


args.use_cuda = args.use_cuda and torch.cuda.is_available()

print("Loading drug dataset...")
# dataset, ptw_ids = load_dataset(input_dir=args.input_dir, repository=args.repository, drug_id=args.drug_id)
dataset, ptw_ids = load_dataset(input_dir=args.input_dir, repository=args.repository, drug_id=args.drug_id, random_seed=SEED, shuffle_feature=args.shuffle)

train_set, test_set = split_dataset(dataset, ratio=0.8)

# replace tgt in train_set
train_set['tgt'], train_set['msk'] = fill_mask(train_set['tgt'], train_set['msk'])

args.exp_size = dataset['exp_bin'].shape[1]
args.mut_size = dataset['mut_bin'].shape[1]
args.cnv_size = dataset['cnv_bin'].shape[1]

if args.omic == 'exp':
  args.omc_size = args.exp_size
elif args.omic == 'mut':
  args.omc_size = args.mut_size
elif args.omic == 'cnv':
  args.omc_size = args.cnv_size

args.drg_size = dataset['tgt'].shape[1]
args.train_size = len(train_set['tmr'])
args.test_size = len(test_set['tmr'])


print("Hyperparameters:")
print(args)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    
    logs = {'args':args, 'iter':[],
            'precision':[], 'recall':[],
            'f1score':[], 'accuracy':[], 'auc':[],
            'precision_train':[], 'recall_train':[],
            'f1score_train':[], 'accuracy_train':[], 'auc_train':[],
            'loss':[], 'ptw_ids':ptw_ids}

    if args.eval:
        print(f"Evaluating from saved logs at: {args.pkl_path}")
        with open(args.output_dir + args.pkl_path, "rb") as f:
            logs = pickle.load(f)
        from utils import evaluate_all

        preds = logs["preds"]
        labels = logs["labels"]
        msks = logs["msks"]
        precision, recall, f1, acc, auc_roc, auc_pr = evaluate_all(labels, msks, preds)

        print(f"\nEvaluation Metrics from {args.pkl_path}:")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"AUC-PR: {auc_pr:.4f}")
        exit()

    model = CF(args)
    print("HIDDEN EMBEDDING SIZE:", args.hidden_dim_enc)
    model.build(ptw_ids)

    if args.use_cuda:
        model = model.cuda()

    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    if args.is_train:
        print("Training...")
        logs = model.train(train_set, test_set,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            max_iter=args.max_iter,
            max_fscore=args.max_fscore,
            test_inc_size=args.test_inc_size,
            logs=logs
        )

        labels, msks, preds, tmr, amtr = model.test(test_set, test_batch_size=args.test_batch_size)
        labels_train, msks_train, preds_train, tmr_train, amtr_train = model.test_train(train_set, test_batch_size=args.test_batch_size)

        logs["preds"] = preds
        logs["msks"] = msks
        logs["labels"] = labels
        logs['tmr'] = tmr
        logs['amtr'] = amtr

        logs['preds_train'] = preds_train
        logs['msks_train'] = msks_train
        logs['labels_train'] = labels_train
        logs['tmr_train'] = tmr_train
        logs['amtr_train'] = amtr_train

    else:
        print("LR finding...")
        logs = model.find_lr(train_set, test_set,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            max_iter=args.max_iter,
            max_fscore=args.max_fscore,
            test_inc_size=args.test_inc_size,
            logs=logs
        )

    if args.store_model:
        if not args.pkl_path:
            for trial in range(100):
                trial_path = os.path.join(args.output_dir, f"logs{trial}.pkl")
                if not os.path.exists(trial_path):
                    print(f"Auto-saving model to {trial_path}")
                    with open(trial_path, "wb") as f:
                        pickle.dump(logs, f, protocol=2)
                    break
        else:
            save_path = os.path.join(args.output_dir, args.pkl_path)
            if os.path.exists(save_path):
                print(f"Warning: Overwriting existing file {save_path}")
            else:
                print(f"Saving model to {save_path}")
            with open(save_path, "wb") as f:
                pickle.dump(logs, f, protocol=2)
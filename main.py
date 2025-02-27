from model import HANLinkPredictor
from load_dataset import load_link_prediction_dataset_folds, global_index

from metrics import roc_auc_score, average_precision_score
from metrics import macro_f1_score

import torch
import json
import os
import gc
import copy


def main(args):
    train_networks1, train_networks2, train_node_pairs, valid_node_pairs, valid_labels, valid_constractive_pairs, test_networks1, test_networks2, test_node_pairs, test_labels, train_node_features, test_node_features = load_link_prediction_dataset_folds(
        args.dataset, args.dataset_path, ratio=args.ratio,
        use_structure_feature=args.feature == 'structure')

    device = torch.device('cuda:{i}'.format(i=args.gpu) if torch.cuda.is_available() else 'cpu')

    node_features_shape = train_node_features[0].shape[1]

    for fold in range(5):
        print(f'Fold {fold + 1}')

        model = HANLinkPredictor(node_features_shape, args.hidden_dim, args.hidden_dim * args.num_heads[-1],
                                 args.num_heads, args.num_heads,
                                 args.dropout,
                                 len(train_networks1[fold]), len(train_networks2[fold]))
        model = model.to(device)

        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        train_networks1_tmp = [g.to(device) for g in train_networks1[fold]]

        train_networks2_tmp = [g.to(device) for g in train_networks2[fold]]

        train_node_features_tmp = train_node_features[fold]

        train_node_features_tmp = train_node_features_tmp.to(device)

        train_node_pairs_tmp = train_node_pairs[fold].to(device)

        train_labels = torch.cat(
            [torch.ones(train_node_pairs_tmp.shape[0] // (args.ratio + 1)),
             torch.zeros(train_node_pairs_tmp.shape[0] // (args.ratio + 1) * args.ratio)],
            dim=0).to(device)

        test_node_pairs_tmp = test_node_pairs[fold]
        test_labels_tmp = test_labels[fold]

        best_state = None

        for epoch in range(args.epochs):
            model.train()

            h1, h2 = model(train_networks1_tmp, train_networks2_tmp, train_node_features_tmp)

            train_output = []
            batch_size = 500
            for i in range(0, train_node_pairs_tmp.shape[0], batch_size):
                end = min(i + batch_size, train_node_pairs_tmp.shape[0])
                train_output.append(
                    model.embed_link(h1[train_node_pairs_tmp[i:end][:, 0]], h2[train_node_pairs_tmp[i:end][:, 1]]))

            train_output = torch.cat(train_output, dim=0)

            current_state = model.state_dict()
            current_state = copy.deepcopy(current_state)

            loss = loss_fn(train_output, train_labels)
            loss.backward()
            optimizer.step()

            model.eval()

            optimizer.zero_grad()

            del train_output

            print(
                f'Epoch {epoch + 1} - Train Loss: {loss.item()}')

            if True:
                best_state = current_state
                early_stopping = 0
                epoch_min = epoch

            if (epoch + 1) % args.epochs == 0:
                with torch.no_grad():
                    test_ouput = []
                    for i in range(0, test_node_pairs_tmp.shape[0], batch_size):
                        end = min(i + batch_size, test_node_pairs_tmp.shape[0])
                        test_ouput.append(model.embed_link(h1[test_node_pairs_tmp[i:end][:, 0]],
                                                           h2[test_node_pairs_tmp[i:end][:, 1]]))

                    test_ouput = torch.cat(test_ouput, dim=0)
                    test_ouput = test_ouput.cpu()

                    test_loss = loss_fn(test_ouput, test_labels_tmp)

                    test_ouput_new = []
                    for i in range(0, test_node_pairs_tmp.shape[0], batch_size):
                        end = min(i + batch_size, test_node_pairs_tmp.shape[0])
                        test_ouput_new.append(torch.sigmoid(test_ouput[i:end]))

                    test_ouput = torch.cat(test_ouput_new, dim=0)

                test_auc = roc_auc_score(test_ouput, test_labels_tmp)
                test_ap = average_precision_score(test_ouput, test_labels_tmp)

                test_macro_f1 = macro_f1_score(test_ouput, test_labels_tmp)

                print(f'Current State -- Test Loss: {test_loss.item()} - Test AUC: {test_auc} - Test AP: {test_ap}')
                print(
                    f'Test Macro F1: {test_macro_f1}')

                del test_ouput, test_ouput_new
            del h1, h2
            gc.collect()
            torch.cuda.empty_cache()

        if best_state is None:
            best_state = current_state
        else:
            print(f'Best Epoch: {epoch_min + 1}')

        print(f'Final State -- Test Loss: {test_loss.item()} - Test AUC: {test_auc} - Test AP: {test_ap}')
        print(
            f'Test Macro F1: {test_macro_f1}')

        os.makedirs(args.results, exist_ok=True)
        with open(os.path.join(args.results, f'fold_{fold + 1}_results.json'), 'w') as f:
            json.dump({'auc': test_auc, 'ap': test_ap,
                       'macro_f1': test_macro_f1
                       }, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='ppi_phenotype',
                        help='Dataset')
    parser.add_argument('--dataset_path', type=str,
                        default='./data',
                        help='Dataset path')
    parser.add_argument('--hidden_dim', type=int, default=6)
    parser.add_argument('--num_heads', type=int, nargs='+',
                        default=[8], help='Number of heads')
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    # parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--results', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--feature', type=str, default='structure',
                        help='Feature type')
    parser.add_argument('--ratio', type=int, default=2,
                        help='Ratio of negative to positive samples')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    import numpy as np

    np.random.seed(args.seed)

    print(args.__dict__)

    main(args)

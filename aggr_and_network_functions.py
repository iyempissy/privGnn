import numpy as np
from six.moves import xrange
import torch
import sys
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import time
from torch.autograd import Variable

from torch_geometric.nn import GCNConv, SAGEConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.utils import subgraph
from torch_sparse import SparseTensor

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')

use_gpu = torch.cuda.is_available()

import all_config as config
config = config.config


def clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    print("parameters", parameters.type())
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = torch.norm(parameters, norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        # for p in parameters:
        #     p.grad.detach().mul_(clip_coef)
        parameters.mul_(clip_coef)
    return parameters


# clip logit and add noise
def clip_logits_norm(parameters, max_norm=4, norm_type=2, epsilon=0.1, add_gaussian_noise=False):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    print("b4 parameters", parameters)
    print("param shape", parameters.shape)

    print("after multiplying parameters", parameters)
    # add Gaussian noise
    if add_gaussian_noise:
        # change to numpy normal distribution. Seems something is odd here
        parameters = parameters.cpu().detach().numpy()
        # Add Gaussian noise to each logits individually to make it differentially private (DP)
        for i in range(0, len(parameters)):
            for j in range(0, len(parameters[i])):
                parameters[i][j] += np.random.normal(scale=0.2)

        # convert back to float tensor
        parameters = torch.FloatTensor(parameters)

    return parameters




# {{{{{{{{{{{{{{{{{{{{{{{{{{{{{{ GNN model }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 2  # 3 cos I removed train_acc from the test
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True, add_self_loops=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 3):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True, add_self_loops=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, in_channels, cached=True, add_self_loops=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True, add_self_loops=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        if isinstance(x, list):  # this will be executed only for train data. Test data is already a tensor
            x = torch.stack(x)

        for i, conv in enumerate(self.convs[:-2]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        features = self.convs[-2](x, adj_t)
        x = self.convs[-1](x, adj_t)

        return features, x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 3):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, in_channels))  # feature_conv
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-2]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        features = self.convs[-2](x, adj_t)

        x = self.convs[-1](x, adj_t)

        return features, x




# MLP

class MLP(nn.Module):
    # define nn
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(config.nb_features, 100)  # normal
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, config.nb_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return X


    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def reset_parameters(m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()


# {{{{{{{{{{{{{{{{{{{{{{{{{{{{{{ Aggregation functions }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
# 1. aggregation_knn

def aggregation_knn(labels, gaussian_scale, return_clean_votes=False):
    """
      This aggregation mechanism takes the label output of knn
      shape is num_query * k, where k from knn
      resulting from inference on identical inputs and computes the most frequent
      label.
      :param labels: logits or probabilities for each sample
      gaussian_scale: sigma for gaussian noise
      :return the most frequent label:
    """
    print('gaussian scale=', gaussian_scale)
    print('labels shape', labels.shape)  # num_queries x num_teachers
    labels_shape = np.shape(labels)
    result = np.zeros(labels_shape[0])
    clean_votes = np.zeros([labels_shape[0], config.nb_labels])

    for i in xrange(int(labels_shape[0])):
        label_count = np.bincount(labels[i, :],
                                  minlength=config.nb_labels)  # count the number of each vote for each label

        clean_votes[i] = np.bincount(labels[i, :], minlength=config.nb_labels)
        for item in xrange(config.nb_labels):  # this is all the labels
            # Add noise to the label count
            label_count[item] += np.random.normal(scale=gaussian_scale)
            label_counts = np.asarray(label_count, dtype=np.float32)

        result[i] = np.argmax(label_counts)  # noisy label counts. This returns the index of the max label_count

    results = np.asarray(result, dtype=np.int32)
    clean_votes = np.array(clean_votes, dtype=np.int32)

    if config.confident == True:
        max_list = np.max(clean_votes, axis=1)  # max vote of each query
        for i in range(len(labels)):
            # add noise using config.sigma1 to the max_list i.e clean vote. Remember clean votes never had noise. 1st time of adding noise
            max_list[i] += np.random.normal(scale=config.sigma1)

        idx_keep = np.where(
            max_list > config.threshold)  # returns index where the max_list > threshold. Dimension is 1 x num_query
        print("idx_keep", idx_keep)
        idx_remain = np.where(max_list < config.threshold)
        release_vote = clean_votes[idx_keep]
        print("release_vote", release_vote)

        # The released vote is just the index of the noisy clean vote that you want to keep.
        # This is like randmoly selecting the clean vote if there is higher concensus.
        # Simply, the noisy screening is that you select the clean vote that has a value > threshold

        confi_result = np.zeros(len(idx_keep[0]))
        for idx, i in enumerate(release_vote):
            # print('release_vote',release_vote[idx])
            for item in range(config.nb_labels):
                # add another noise to the released vote (2nd noise) using config.gau_scale
                release_vote[idx, item] += np.random.normal(scale=config.gau_scale)
            confi_result[idx] = np.argmax(release_vote[idx])  # returns the index of the max vote
        confi_result = np.asarray(confi_result, dtype=np.int32)
        return idx_keep, confi_result, idx_remain

    idx_keep = np.where(results > 0)
    return idx_keep, results

    if return_clean_votes:
        # Returns several array, which are later saved:
        # result: labels obtained from the noisy aggregation
        # clean_votes: the number of teacher votes assigned to each sample and class
        # labels: the labels assigned by teachers (before the noisy aggregation)
        return result, clean_votes, labels
    else:
        # Only return labels resulting from noisy aggregation
        return result


# {{{{{{{{{{{{{{{{{{{{{{{{{{{{{{ Network functions }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
# 1. pred
# 2. train_each_teacher. It depends on train and test function

@torch.no_grad()
def pred(model, data_x, data_y, edge_index, evaluator, save_path, return_feature=False):
    '''
    This function is for updating the teachers with features from student. It is used in extract features function in knn_graph
    1. The pred() for private data will have a dummy edge index while the pred() for public data will have full edge index in
    extract_feature!

    :param model: model
    :param data_x: features
    :param edge_index: edge index
    :param save_path: path to save the trained model
    :param return_feature: if true return feature with the prediction
    :param test_idx: optional. The test index to slice
    :return: feature_list and predictions or predictions alone
    '''

    # print("edge_idx shapppppe", edge_index.shape)
    # print("data_x.shape", data_x.shape)
    # isPublic uses test index to slice through the model
    # This function loads the model saved in save_path and use it aggregation_knnfor prediction "data"


    if use_gpu:
        print("Currently using GPU {}".format(config.gpu_devices))
    else:
        print("Currently using CPU (GPU is highly recommended)")

    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        pred_list, feature_list = [], []
        float_logit_list = []

        features, logits = model(data_x, edge_index)
        predA = logits.log_softmax(dim=-1)

        predA = predA.cpu()
        float_logit_list.append(torch.sigmoid(predA))
        predA = predA.argmax(dim=-1, keepdim=True)


        pred_acc = evaluator.eval({
            'y_true': data_y,
            # This was selected by the association of all teachers. Therefore, no need to slice
            'y_pred': predA,
        })['acc']

        print("Pred Acc after update", pred_acc)
        if return_feature is True:
            feature_list.append(features.cpu())
        pred_list.append(predA)
    predA_t = (((torch.cat(pred_list,
                           0)).float()).numpy()).tolist()  # Its's fine: only 1 list not multiple in the pred_list since not using loader / batching
    predA_t = np.array(predA_t)

    if return_feature == True:
        feature_list = (((torch.cat(feature_list, 0)).float()).numpy()).tolist()
        feature_list = np.array(feature_list)
        feature_list = torch.FloatTensor(feature_list)
        return feature_list, predA_t
    else:
        return predA_t


# using the data of the student to update the features of the teachers!. The goal is to save the model trained on student data

@torch.enable_grad()
def train_each_teacher(model, optimizer, num_epoch, public_x, public_label, public_label_from_teachers,
                       public_edge_index,
                       public_train_idx, public_test_idx, evaluator, save_path, isFeature):
    print('train_data', len(public_x[public_train_idx]), 'public_train_label', len(public_label_from_teachers))

    print("==> Start training")

    start_time = time.time()
    for epoch in range(num_epoch):

        print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, 0.01))
        '''
        We are training the student model on the public data. This is like the Hog flag if the public_train_x is the feature directly from the data else feature flag 
        Hog is synonymous to using the direct features from the data.
        We will use the saved model to get features for the private model (this is the feature flag)
        '''

        train(model, optimizer, public_train_idx, public_x, public_label_from_teachers, public_edge_index,
              evaluator, epoch)

        test(model, public_test_idx, public_x, public_label, public_edge_index, evaluator,
             isFeature=isFeature)

    _, final_test_acc = test(model, public_test_idx, public_x, public_label, public_edge_index, evaluator,
                          isFeature=isFeature)

    # save model
    if use_gpu:
        state_dict = model.state_dict()
    else:
        state_dict = model.state_dict()
    print('save model', save_path)
    torch.save(state_dict, save_path)

    elapsed = round(time.time() - start_time)

    print("Finished. Training time (h:m:s): {}.".format(elapsed))
    return final_test_acc


@torch.enable_grad()
def train(model, optimizer, train_idx, public_x, public_label_from_teachers, public_edge_index, evaluator, epoch,
          isInductive=True, isBaseline=False):
    ''' For training the public train data such that it returns trained features!
    It is used in train_each_teacher function
    Note: public_label_from_teachers becomes the groundtruth for the selected public_y cos it is the one selected by all teachers
    '''

    model.train()
    public_train_y = torch.LongTensor(public_label_from_teachers)
    optimizer.zero_grad()

    if isinstance(public_x, list):  # this will be executed only for train data. Test data is already a tensor
        public_x = torch.stack(public_x)

    # have tensors of all zeros n set to 1 if its in train_idx
    fill = np.zeros(public_x.shape[0])
    fill[train_idx] = 1
    fill = torch.LongTensor(fill)
    train_mask = fill > 0

    feature, logits = model(public_x,
                         public_edge_index)

    out = logits.log_softmax(dim=-1)

    loss = F.nll_loss(out[train_idx], public_train_y.squeeze(0).to(device))
    y_pred = out.argmax(dim=-1, keepdim=True)

    loss.backward()
    optimizer.step()

    train_acc = evaluator.eval({
        'y_true': public_train_y.unsqueeze(1),
        # This was selected by the association of all teachers. Therefore, no need to slice
        'y_pred': y_pred[train_idx],
    })['acc']

    print(f'Epoch: {epoch:02d}, '
          f'Loss: {loss.item():.4f}, '
          f'Public (train): {100 * train_acc:.2f}%')

    return feature, train_acc


@torch.no_grad()
def test(model, test_idx, public_x, public_y, public_edge_index, evaluator, isInductive=True, isBaseline=False,
         isFeature=False):
    model.eval()
    public_y = public_y.cpu()
    test_idx = test_idx.cpu()

    public_data_y = torch.LongTensor(public_y)  # Remaining  public data

    feature, logits = model(public_x, public_edge_index)

    out = logits.log_softmax(dim=-1)

    y_pred = out.argmax(dim=-1, keepdim=True)


    test_acc = evaluator.eval({
        'y_true': public_data_y[test_idx],  # for test data
        'y_pred': y_pred[test_idx],  # do slicing
    })['acc']

    print(f'PrivateAccuracy (test): {100 * test_acc:.2f}%')
    return feature, test_acc



@torch.enable_grad()
def train_baseline(model, optimizer, train_idx, data_x, data_y, edge_index, evaluator, isBaseline1=False):

    # For train_baseline(), when isBaseline1=True implies that the loss and the accuracy is computed over all the y
    # if isBaseline1=False, the loss and accuracy is computed over only the selected or sliced y

    model.train()
    # print("train_idx", train_idx.shape)
    data_y = data_y.cpu()
    data_y = torch.LongTensor(data_y)
    optimizer.zero_grad()

    if isinstance(data_x, list):  # this will be executed only for train data. Test data is already a tensor
        data_x = torch.stack(data_x)

    data_x = data_x.to(device)

    feature, logits = model(data_x, edge_index)

    out = logits.log_softmax(dim=-1)

    if isBaseline1:
        # For private data, no need to slice cos inductive
        loss = F.nll_loss(out, data_y.squeeze(1).to(device))
    else:
        # for public data, we slice cos we are in transductive
        loss = F.nll_loss(out[train_idx], data_y.squeeze(1)[train_idx].to(device))

    y_pred = out.argmax(dim=-1, keepdim=True)

    loss.backward()
    optimizer.step()

    if isBaseline1:
        train_acc = evaluator.eval({
            'y_true': data_y,
            'y_pred': y_pred,
        })['acc']
    else:
        train_acc = evaluator.eval({
            'y_true': data_y[train_idx],
            'y_pred': y_pred[train_idx],
        })['acc']

    if not config.is_tkt: # no need to print for tkt cos u will train each model for each query for xxx epochs. This will be too much to print
        print(f'Baseline (train): {100 * train_acc:.2f}%')
    return feature, train_acc



@torch.no_grad()
def test_baseline(model, test_idx, data_x, data_y, edge_index, evaluator, isBaseline1=False, addNoisetoPosterior=False, max_norm_logits=1):

    # For test_baseline(), when isBaseline1=True implies that the accuracy is computed over all the y
    # if isBaseline1=False, the accuracy is computed over only the selected or sliced y

    model.eval()

    data_y = data_y.cpu()
    test_idx = test_idx.cpu()

    data_y = torch.LongTensor(data_y)

    feature, logits = model(data_x, edge_index)

    y_pred = logits.argmax(dim=-1, keepdim=True)
    test_acc = evaluator.eval({
        'y_true': data_y[test_idx],  # for test data
        'y_pred': y_pred[test_idx],  # do slicing
    })['acc']

    print("Test accuracy of logit b4 clipping", test_acc)

    out = logits.log_softmax(dim=-1)





    # Noisy posteriors
    if addNoisetoPosterior:
        out = out.cpu().detach().numpy()
        # Add Laplacian noise to each posterior individually to make it differentially private (DP)
        # I'm adding noise to the posterior here because I'm also using this for the getting the noisy labels for training the student
        beta = 1 / config.epsilon #num_classes or posteriors e.41 cos the sensitivity is not 1 but num_posteriors. New. It should be 2. See the triangle law
        for i in range(0, len(out)):
            # print("out[i][j] b4 noise", out[i])
            for j in range(0, len(out[i])):
                if config.use_lap_noise:
                    out[i][j] += np.random.laplace(0, beta, 1)
                else:
                    # Gaussian noise
                    gau_scale = np.sqrt(2 * np.log(1.25 / config.delta)) * 1 / config.epsilon
                    out[i][j] += np.random.normal(scale=gau_scale)

        # convert back to float tensor
        out = torch.FloatTensor(out)
        y_pred = out.argmax(dim=-1, keepdim=True)
    else:
        y_pred = out.argmax(dim=-1, keepdim=True)


    test_acc = evaluator.eval({
        'y_true': data_y[test_idx],  # for test data
        'y_pred': y_pred[test_idx],  # do slicing
    })['acc']

    print("Test accuracy of clipped logit", test_acc) #==>Same as below. Just printed


    if isBaseline1:
        # private data, no slicing
        test_acc = evaluator.eval({
            'y_true': data_y,  # for test data
            'y_pred': y_pred,  # do slicing
        })['acc']

        final_ypred = y_pred
    else:
        test_acc = evaluator.eval({
            'y_true': data_y[test_idx],  # for test data
            'y_pred': y_pred[test_idx],  # do slicing
        })['acc']
        final_ypred = y_pred[test_idx]

    return test_acc, final_ypred



def train_test_baselines(model, optimizer, num_epoch, train_idx, test_idx, public_data_x, public_data_y,
                         public_edge_index, private_data_x, private_data_y, private_edge_index, evaluator,
                         isBaseline1=False, isBaseline1_star=False):
    # baseline1 is the train on private and test on test data_x
    # baseline2 is train on public and test on public test

    # train_idx and test_idx are for public_train and public_test respectively


    # For train_baseline(), when isBaseline1=True implies that the loss and the accuracy is computed over all the y
    # if isBaseline1=False, the loss and accuracy is computed over only the selected or sliced y
    #
    # For test_baseline(), when isBaseline1=True implies that the accuracy is computed over all the y
    # if isBaseline1=False, the accuracy is computed over only the selected or sliced y

    # Training
    for epoch in range(num_epoch):
        if isBaseline1:
            if isBaseline1_star:
                # For the private data, train_idx is useless (Baseline1_star: Inductive)
                train_baseline(model, optimizer, train_idx, private_data_x, private_data_y, private_edge_index, evaluator,
                               isBaseline1)
            else:
                # Here is the default or original Baseline1 (Inductive). similarly, train_idx is useless.
                # Redefining things here is redundant but it's okay
                train_baseline(model, optimizer, train_idx, private_data_x, private_data_y, private_edge_index,
                               evaluator, isBaseline1)
        else:
            # train_idx is needed here so as to slice only the public_train (Baseline2: Transductive)
            train_baseline(model, optimizer, train_idx, public_data_x, public_data_y, public_edge_index, evaluator,
                           isBaseline1)



    # Testing
    if isBaseline1:
        if isBaseline1_star:
            # The 1st model here is the model of the previously trained model
            # The param of isBaseline1 is set to False here because we are testing in "transductive" i.e on the public_train to 1st obtain no-noisy and noisy versions of train.
            # cos the goal here is to is to use the noisy labels of the public_train obtained from model trained on the private_data to train a GNN model, then use the result to test on the public_test
            # The goal of the isBaseline1=False here is to do the slicing based on the train_idx given. Here is the train_idx
            baseline_test, non_noisy_labels = test_baseline(model, train_idx, public_data_x, public_data_y,
                                                        public_edge_index,
                                                        evaluator, isBaseline1=False, addNoisetoPosterior=False)

            print("Non-noisy version of public_train acc", baseline_test)

            baseline_test, noisy_labels = test_baseline(model, train_idx, public_data_x, public_data_y, public_edge_index,
                                             evaluator, isBaseline1=False, addNoisetoPosterior=True) #add noise to the posteriors of the retireved public train by setting to last parameter to True

            print("Noisy version", baseline_test)
            # This will require 2 models. One for getting the posteriors which was trained earlier. The other for training the public data n testing accordingly
            # reset the model so you can use it to train noisy public data
            model.reset_parameters()

            for epoch in range(num_epoch):
                # train the public train using the noisy labels
                train_baseline(model, optimizer, train_idx, public_data_x, noisy_labels, public_edge_index, evaluator,
                               isBaseline1=False) # isBaseline1 to be set to False cos we don't have a seperate subgraph for the public data. Therefore, train_idx is needed here for slicing

            # test on public test
            baseline_test, _ = test_baseline(model, test_idx, public_data_x, public_data_y, public_edge_index,
                                                evaluator, isBaseline1=False, addNoisetoPosterior=False) # isBaseline1=False cos of slicing and addNoisetoPosterior=False cos no noised is needed


        else:
            # Original Baseline1
            # baseline_test, _ = test_baseline(model, test_idx, private_data_x, private_data_y, private_edge_index, evaluator,
            #                               isBaseline1) ==> Delete this
            # Actually, test_baseline should not have isBaseline1 cos we are only using the information of the public data. Therefore, set to False i.e inductive
            baseline_test, _ = test_baseline(model, test_idx, public_data_x, public_data_y, public_edge_index, evaluator,
                          isBaseline1=False, addNoisetoPosterior=False) # test directly on public_test
    else:
        # Baseline2
        baseline_test, _ = test_baseline(model, test_idx, public_data_x, public_data_y, public_edge_index, evaluator,
                                      isBaseline1) #cos this isBaseline1 will be false anyways. Thus baseline2 cos the test_idx is used in slicing

    return baseline_test

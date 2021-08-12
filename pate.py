import torch
import torch.nn.functional as F
import numpy as np
import aggr_and_network_functions

import all_config as config
config = config.config

import torch.nn as nn

# This is the file for implementing PATE. We will do noisy voting etc. here.

# Let us begin

# Input:  private_data_x, private_data_y, private_data_edge_index, public_data_x, public_data_y, public_data_edge_index, idx_train_public, idx_test_public
# But the following are list of tensors. So we need slice until config.num_teacher_graphs private_data_x, private_data_y, private_data_edge_index,

@torch.enable_grad()
def train_pate(model, optimizer, epochs, data_x, data_y, edge_index, evaluator, device, use_mlp = False):
    # This function is for training the private model. Used in train_models()
    # use_mlp uses mlp instead of GNN for training the private model

    data_y = data_y.cpu()
    data_y = torch.LongTensor(data_y)
    data_x = data_x.to(device)

    feature = 0
    train_acc = 0
    y_pred = 0
    running_loss = 0
    for e in range(epochs):
        model.train()
        # print("train_idx", train_idx.shape)
        optimizer.zero_grad()

        if use_mlp:
            logits = model(data_x) #mlp
            out = logits.log_softmax(dim=-1)
        else:
            feature, logits = model(data_x, edge_index)
            out = logits.log_softmax(dim=-1)
        # For private data, no need to slice cos inductive
        loss = F.nll_loss(out, data_y.squeeze(1).to(device))

        y_pred = out.argmax(dim=-1, keepdim=True)
        # y_pred = y_pred.cpu()

        loss.backward()
        optimizer.step()

        running_loss+=loss.item()

    train_acc = evaluator.eval({
        'y_true': data_y,
        'y_pred': y_pred,
    })['acc']

    print(f'Accuracy with Private training node for each graph: {100 * train_acc:.2f}%')

    return model
    # return feature, train_acc

@torch.no_grad()
def predict_pate(model, data_x, data_y, edge_index, train_idx, evaluator, device, is_stdnt_train=True, use_mlp=False):
    # is_stdnt_train implies that you are predicting over the student train idx that u wanna label by the teachers
    # when it is false, it can be used to predict on the test set and the train_idx becomes the test_idx

    data_y = data_y.cpu()
    data_y = torch.LongTensor(data_y)
    data_x = data_x.to(device)

    model.eval()
    if use_mlp:
        logits = model(data_x) #mlp
        out = logits.log_softmax(dim=-1)
    else:
        feature, logits = model(data_x, edge_index)
        out = logits.log_softmax(dim=-1)

    y_pred = out.argmax(dim=-1, keepdim=True)
    # y_pred = y_pred.cpu()

    test_acc = evaluator.eval({
        'y_true': data_y[train_idx],
        'y_pred': y_pred[train_idx],
    })['acc']

    if is_stdnt_train:
        baba = "nothing" # no need to print this.
        # print(f'PATE Testing with Public stdnt_train_idx (train Acc): {100 * test_acc:.2f}%')
    else:
        print(f'PATE Testing with Public stdnt_test_idx (test Acc) Final: {100 * test_acc:.2f}%')

    final_y_pred = y_pred[train_idx]

    return final_y_pred, test_acc


@torch.enable_grad()
def train_models(num_teacher_graphs, epochs, private_data_x_all, private_data_edge_index_all, private_data_y_all, evaluator, device):
    """ Trains *num_teacher* models (num_teachers being the number of teacher classifiers) """
    models = []
    for i in range(num_teacher_graphs):
        # create new instances everytime. Better than resetting

        if config.use_mlp: #If true, teacher model = MLP else teacher model is GNN
            model = aggr_and_network_functions.MLP().to(device) #using mlp as the private node
        else:
            # normal. using graph for training private
            model = aggr_and_network_functions.SAGE(config.nb_features, config.hidden_channels, config.nb_labels, config.num_layers,
                                                      config.dropout).to(device)  # hardcoded
        # criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        # train(model, teacher_loaders[i], criterion, optimizer)
        model = train_pate(model, optimizer, epochs, private_data_x_all[i], private_data_y_all[i], private_data_edge_index_all[i], evaluator, device, use_mlp = config.use_mlp)
        models.append(model)

        model.reset_parameters()
    return models

def aggregated_teachers(models, public_data_x, public_data_y, public_data_edge_index, stdnt_train_idx, num_classes, epsilon, evaluator, device):
    """ Take predictions from individual teacher model and
        creates the true labels for the student after adding
        laplacian noise to them

        public_data_y ==> used for computing accuracy
        stdnt_train_idx ==> used for slicing the public data to obtain student data
    """
    # We are passing all public_x and public_data_y because we are in transductive. We eill use stdnt_train_idx to do the slicing

    # preds is the prediction of each of the teachers
    preds = torch.zeros((len(models), len(stdnt_train_idx)), dtype=torch.long) # shape num_teachers x len_query
    # Try to unsqueeze preds so we can directly input y in the column format
    preds = preds.unsqueeze(2)
    print("preds shape", preds.shape)
    for i, model in enumerate(models):
        # This is for getting the predictions on public train data that we want the teacher to label i,e student label.
        print("config.use_mlp", config.use_mlp)
        # if config.use_mlp = True, the teaxher model will be MLP, else GNN
        results, train_acc = predict_pate(model, public_data_x, public_data_y, public_data_edge_index, stdnt_train_idx, evaluator, device, use_mlp = config.use_mlp)
        print("Overall Train Accuracy for teacher ", i, " using Public stdnt_train_idx ==>", train_acc)
        # print("results shape", results.shape)
        preds[i] = results

    # squeeze it out
    preds = preds.squeeze(2)

    labels = np.array([]).astype(int)
    for node_preds in np.transpose(preds):
        label_counts = np.bincount(node_preds, minlength=num_classes)
        beta = 1/ epsilon #sensitivity is 1 cos its vote count

        for i in range(len(label_counts)):
            label_counts[i] += np.random.laplace(0, beta, 1)  # add noise

        new_label = np.argmax(label_counts)
        labels = np.append(labels, new_label)

    return preds.numpy(), labels

# Train student using the noisy predictions from teachers then print
# The same as train in aggr_and_network_functions.py file
# This is just a copy of that and this will use the epoch info to train the student


@torch.enable_grad()
def train_student_pate(model, optimizer, train_idx, public_x, public_label_from_teachers, public_edge_index, evaluator, epoch,
          device):
    ''' For training the public train data such that it returns trained features!
    It is used for traiining the train data with noisy labels from the teachers
    Note: public_label_from_teachers becomes the groundtruth for the selected public_y cos it is the one selected by all teachers
    '''

    model.train()
    public_train_y = torch.LongTensor(public_label_from_teachers)

    if isinstance(public_x, list):  # this will be executed only for train data. Test data is already a tensor
        public_x = torch.stack(public_x)

    # have tensors of all zeros n set to 1 if its in train_idx. Then use it accordingly?
    fill = np.zeros(public_x.shape[0])
    fill[train_idx] = 1
    fill = torch.LongTensor(fill)
    train_mask = fill > 0

    feature = 0
    train_acc = 0

    for e in range(epoch):
        optimizer.zero_grad()
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

        print(f'Epoch: {e:02d}, '
              f'Loss: {loss.item():.4f}, '
              f'Accuracy of training Public train data with noisy labels(stdnt_train_idx): {100 * train_acc:.2f}%')

    # return loss.item(), train_acc
    return feature, train_acc

def pate_graph(private_data_x, private_data_y, private_data_edge_index, public_data_x, public_data_y, public_data_edge_index, idx_train_public, idx_test_public):
    print("I am here")
    return "Done"
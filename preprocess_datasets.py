import time
import torch
import random
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from torch_sparse import SparseTensor
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import all_config as config
config = config.config


def get_inductive_spilt(data, num_classes, num_train_Train_per_class, num_public_train, num_public_test, rand_state, data_name):
    # -----------------------------------------------------------------------
    # target_train, target_out
    # shadow_train, shadow_out
    '''
    Randomly choose 'num_train_Train_per_class' and 'num_train_Shadow_per_class' per classes for training Target and shadow models respectively
    Random choose 'num_public_train' and 'num_public_test' for testing (out data) Target and shadow models respectively

    '''
    overall_start_time = time.time()

    # convert all label to list
    label_idx = data.y.cpu().detach().numpy().tolist()
    print("label_idx", len(label_idx))
    private_train_idx = []
    # public_train_idx = []
    # public_test_idx = []

    # for i in range(num_classes):
    #     c = [x for x in range(len(label_idx)) if label_idx[x] == i]
    #     print("c", len(c)) #the min is 180 which is 7th class
    #     sample = random.sample(range(c),num_train_Train_per_class)
    #     private_train_idx.extend(sample)

    if config.is_arxiv_random_split:
        data.y = data.y.squeeze(1)

    for c in range(num_classes):

        idx = (data.y == c).nonzero().view(-1)
        sample_train_idx = idx[torch.randperm(idx.size(0))]
        sample_private_train_idx = sample_train_idx[:num_train_Train_per_class]
        private_train_idx.extend(sample_private_train_idx)

        print("idx.size(0)", idx.size(0))  # this is the total number of data in each class

    print("private_train_idx", len(private_train_idx))

    # Randomly reshuffle private_train_idx
    random.shuffle(private_train_idx)

    # TODO in the next round, make public train also based on class splitting

    print()
    # check if file exist else do the running again. This saves time
    if os.path.isfile(config.save_model + "/public_train_idx" + data_name + "_" + str(num_public_train) + "_" + str(rand_state) + ".pt"):
        public_train_idx = torch.load(config.save_model + "/public_train_idx" + data_name + "_" + str(num_public_train) + "_" + str(rand_state) + ".pt")
        public_test_idx = torch.load(config.save_model + "/public_test_idx" + data_name + "_" + str(num_public_test) + "_" + str(rand_state) + ".pt")

    else:
        # run again
        public_train_start_time = time.time()
        others = [x for x in range(len(label_idx)) if x not in set(private_train_idx)]
        # print("others",others)
        print("done others")
        public_train_idx = torch.LongTensor(random.sample(others, num_public_train))
        # save train_idx since it takes long
        torch.save(public_train_idx, config.save_model + "/public_train_idx" + data_name + "_" + str(num_public_train) + "_" + str(rand_state) + ".pt")
        public_train_end_time = time.time()
        print("done public train time", public_train_end_time - public_train_start_time)

        public_test_start_time = time.time()
        public_test = [x for x in others if x not in set(public_train_idx)]
        public_test_idx = torch.LongTensor(random.sample(public_test, num_public_test))
        # save test_idx
        torch.save(public_test_idx, config.save_model + "/public_test_idx" + data_name + "_" + str(num_public_test) + "_" + str(rand_state) + ".pt")
        public_test_end_time = time.time()
        print("done public test time", public_test_end_time - public_test_start_time)

    print("public_train_idx", len(public_train_idx))
    print("public_test_idx", len(public_test_idx))

    # ----------set values for mask--------------------------------
    # Also changed this so as to conform with the shape of others. No, this uncommented one is better
    # We don't need this cos we have already created a subgraph outta it

    private_train_mask = torch.ones(len(private_train_idx),
                                    dtype=torch.uint8)  # although useless we use all data anyways

    # private_train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
    # for i in private_train_idx:
    #     private_train_mask[i] = 1

    # ---test-mask---
    public_train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
    for i in public_train_idx:
        public_train_mask[i] = 1
    # ---val-mask-----
    public_test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
    for i in public_test_idx:
        public_test_mask[i] = 1

    '''
    get all nodes and corresponding edge_index information
    '''
    # This is for creating subgraphs

    # For target

    # train
    private_data_x = data.x[private_train_idx]

    # Begin TODO N
    # tt = private_train_idx[0]
    # print("tt", tt) #tensor(1110)
    # torch.set_printoptions(threshold=10000)
    # print("target_x_inductiveInitial", private_x)
    # print("data.x[1110]", data.x[1110]) # private_train_idx[0]
    # print("private_x[0]", private_x[0]) # equivalent to data.x[1110] cos 1110 is the 1st element of private_train_idx index list
    # print("data.x[1395]", data.x[1395])

    '''# print("private_x[1110]", private_x[1110]) # This will throw error cos the x i.e private_x starts from 0 and ends at 630. Simply idx are reassigned forming a kinda new graph
    # So the original node ID is reassigned a new node ID with number ranging from 0 to 630
    # Rather, do this:'''
    # if torch.all(torch.eq(data.x[tt], private_x[0])):
    #     print("private_x[0]", private_x[0])
    #     print("private_x[1]", private_x[1])

    # End TODO N

    private_data_y = data.y[private_train_idx]

    # Need to unsqueeze so as to conform with arxiv
    private_data_y = private_data_y.unsqueeze(1)
    private_data_edge_index, _ = subgraph(private_train_idx, data.edge_index, relabel_nodes=True)

    print("private_data_x", private_data_x.shape)
    print("private_data_y", private_data_y.shape)

    # public train
    public_train_x = data.x[public_train_idx]
    public_train_y = data.y[public_train_idx]

    # public test
    public_test_x = data.x[public_test_idx]
    public_test_y = data.y[public_test_idx]

    # all public data
    public_data_x = torch.cat((public_train_x, public_test_x), 0)
    public_data_y = torch.cat((public_train_y, public_test_y), 0)

    # Need to unsqueeze so as to conform with arxiv
    public_data_y = public_data_y.unsqueeze(1)

    print("public_data_x", public_data_x.shape)
    print("public_data_y", public_data_y.shape)

    # get new indexes for the train and test. This allows for the right slicing when testing!
    idx_train_public = torch.LongTensor([x for x in range(0, len(public_train_idx))])
    idx_test_public = torch.LongTensor(
        [x for x in range(len(public_train_idx), len(public_train_idx) + len(public_test_idx))])

    # add both train n test public index together
    all_public_idx = torch.cat((public_train_idx, public_test_idx), 0)
    public_data_edge_index, _ = subgraph(all_public_idx, data.edge_index, relabel_nodes=True)

    # data = Data(private_data_x=private_data_x, private_data_edge_index=private_data_edge_index,
    #             private_data_y=private_data_y,
    #             public_data_x=public_data_x, public_data_edge_index=public_data_edge_index,
    #             public_data_y=public_data_y,
    #             private_train_mask=private_train_mask, idx_train_public=idx_train_public,
    #             idx_test_public=idx_test_public)
    overall_end_time = time.time()
    print("Total time: ", overall_end_time - overall_start_time)
    # return data

    # if we wanna return data, then we need do
    # amazon_graph = get_inductive_spilt(data, config.nb_labels, 250, 3000, 3000)
    # private_data_x = amazon_graph.private_data_x
    # private_data_y = amazon_graph.private_data_y
    # private_data_edge_index = amazon_graph.private_data_edge_index
    # public_data_x = amazon_graph.public_data_x
    # public_data_y = amazon_graph.public_data_y
    # # # public_data_y needs to be converted to cpu? Seem this is the solution to the error of line : public_test_labels = np.array(public_data_test_y)
    # # public_data_y = public_data_y.cpu()
    # public_data_edge_index = amazon_graph.public_data_edge_index
    # idx_train_public = amazon_graph.idx_train_public
    # idx_test_public = amazon_graph.idx_test_public
    # private_idx = amazon_graph.private_train_mask

    print("private_data_edge_index", private_data_edge_index.shape)
    print("public_data_edge_index", public_data_edge_index.shape)

    # Return directly
    return private_data_x, private_data_y, private_data_edge_index, public_data_x, public_data_y, public_data_edge_index, idx_train_public, idx_test_public





def get_preprocessed_arxiv_dataset(dataset, use_sparse, device):
    # data = dataset[0]
    # data = data.to(device)
    # all_edge_index = SparseTensor.from_edge_index(data.edge_index)  # This is equivalent to doing transform
    # print("all_edge_index", all_edge_index)
    #
    # all_edge_index = all_edge_index.to_symmetric()
    #
    # print("all_edge_index symmetric", all_edge_index)

    split_idx = dataset.get_idx_split()

    train_idx, test_idx, val_idx = split_idx["train"], split_idx["test"], split_idx["valid"]

    private_idx, public_train_idx, public_test_idx = train_idx, test_idx, val_idx
    # print("private_idx print", private_idx)

    graph = dataset[0].to(device)

    # print("No of graphs", len(dataset))
    # print("graph", graph)
    # print("num_nodes", graph.num_nodes)
    # print("num_node_features", graph.num_node_features)
    # print("num_edges", graph.num_edges)
    # print("num_edge_features", graph.num_edge_features)
    # print("Num classes", dataset.num_classes)
    # print("Average node degree", graph.num_edges / graph.num_nodes)
    # print("No of training nodes", train_idx.shape)
    # print("Training nodes label rate", len(train_idx) / graph.num_nodes)
    # print("Contains isolated nodes", graph.contains_isolated_nodes())
    # print("Contains self-loops", graph.contains_self_loops())
    # print("Is undirected", graph.is_undirected())

    # # Seeing what the trainnode features look like for 1 node and print its features vectors
    # print("Train node features of all node", graph.x[train_idx])
    # print("Train node features of 1 node", graph.x[train_idx][0])
    # print("Label", graph.y[train_idx])
    # print("train_idx", train_idx.shape)
    # print("test_idx", test_idx.shape)
    # print("Val_idx", val_idx.shape)

    # Do transductive for public i.e combine test_val_idx
    all_public_idx = torch.cat((public_train_idx, public_test_idx), 0)

    # print("all_public_idx", all_public_idx.shape)
    # print("All graph edge_index", graph.edge_index)
    #
    # print("\n\n\n ========================================================================================== \n")

    # create 1 graph (subgraph) from train idx and call it private data.
    # Also create 2 graphs from test and validation index and call it public_train and public_test respectively

    # subgraph returns the edge index. edge_attr that only contains info about the train data
    private_data_subgraph = subgraph(private_idx, edge_index=graph.edge_index, relabel_nodes=True)
    print("private_data subgragh", private_data_subgraph)
    private_data_edge_index = private_data_subgraph[0]

    # To use
    # graph.x[private_idx] to get node features and private_data_edge_index as the edge_index

    # uses default as the hog
    private_data_x = graph.x[private_idx]
    private_data_y = graph.y[private_idx]

    print("private_data_x", private_data_x.shape)
    print("private_data edge index", private_data_edge_index.shape)  # get only edge_index
    print("Private_data_average degree", private_data_edge_index.size(1) / private_data_x.size(0))

    # public_data_train_subgraph = subgraph(test_idx, edge_index=graph.edge_index, relabel_nodes=True)
    # public_data_train_edge_index = public_data_train_subgraph[0]

    # print("public_data_train_edge_index", public_data_train_edge_index)

    # public_data_test_subgraph = subgraph(val_idx, edge_index = graph.edge_index, relabel_nodes=True)
    # public_data_test_edge_index = public_data_test_subgraph[0]

    # Don't split test and val (public train and test). Make them one graph and only slice through them as in transductive.
    # The only thing that needs to be distinct is that of the private vs public

    print("all_public_idx.shape[0]", all_public_idx.shape[0])
    public_data_subgraph = subgraph(all_public_idx, edge_index=graph.edge_index,
                                    relabel_nodes=True)  # the num_nodes is the one causing problem. Need to force to be the same as that if all_public_idx
    public_data_edge_index = public_data_subgraph[0]

    print("public_data_edge_index", public_data_edge_index)
    print("Empty public edge_index", public_data_edge_index[:, :0])

    public_data_train_x = graph.x[public_train_idx]
    public_data_train_y = graph.y[
        public_train_idx]  # this is not needed as we will label the public train with data from private data. No use this as the baseline

    public_data_test_x = graph.x[public_test_idx]
    public_data_test_y = graph.y[public_test_idx]

    # concatenate them together so you can use when you use sparse tensor. Uncomment if not sparse tensor
    # This ensures that everything is preserved!
    public_data_x = torch.cat((public_data_train_x, public_data_test_x), 0)
    public_data_y = torch.cat((public_data_train_y, public_data_test_y), 0)

    print("public_data_x", public_data_x, public_data_x.shape)
    print("public_data_y", public_data_y, public_data_y.shape)

    # get new indexes for the train and test. This allows for the right slicing when testing!
    idx_train_public = torch.LongTensor([x for x in range(0, len(public_train_idx))])
    idx_test_public = torch.LongTensor([x for x in range(len(public_train_idx), len(public_train_idx) + len(public_test_idx))])

    # print("This public_data_y[78401]", public_data_y[78401])

    print("idx_train_public", idx_train_public, idx_train_public.shape)

    print("idx_test_public", idx_test_public, idx_test_public.shape)

    print("Public edge index", public_data_edge_index.shape)

    print("public_data_train_x", public_data_train_x.shape)
    print("Public_data_average train degree", public_data_edge_index.size(1) / public_data_train_x.size(0))
    print("public_data_test_x", public_data_test_x.shape)
    print("Public_data_average test degree", public_data_edge_index.size(1) / public_data_test_x.size(0))

    print("private_data_xprivate_data_x before", private_data_x)
    print("public_data_train_xpublic_data_train_x before", public_data_train_x)

    if use_sparse:
        # Transform and "symmetricize" the edge indexes for private and public.
        # Symmetric performs better than using edge index from the PyG author's code

        private_data_edge_index = SparseTensor.from_edge_index(private_data_edge_index)
        private_data_edge_index = private_data_edge_index.to_symmetric()

        # print("private_data_edge_index symm", private_data_edge_index)

        public_data_edge_index = SparseTensor.from_edge_index(public_data_edge_index, sparse_sizes=(
            all_public_idx.shape[0], all_public_idx.shape[0]))  # solution to size mismatch is specify sparse_sizes
        # print("public_data_edge_index b4", public_data_edge_index)
        public_data_edge_index = public_data_edge_index.to_symmetric()

        # print("public_data_edge_index symmetric", public_data_edge_index)
        #
        # print("public_data_edge_index2", public_data_edge_index.size(0))

    return private_data_x, private_data_y, private_data_edge_index, public_data_x, public_data_y, public_data_edge_index, idx_train_public, idx_test_public




def plot_labels(labels, num_answered_query=0, total_num_queries=0, is_all_test_labels=False, is_correct_label=False): # set is_all_test_labels to true for all test lebel. Set is correct_label to true for correctly released label else released data
    unique, counts = np.unique(labels, return_counts=True)
    labels_stat = dict(zip(unique, counts))
    total_num_labels = sum(labels_stat.values())

    # if the label is not in released vote, set it to 0
    for key in range(0, config.nb_labels):
        if key not in labels_stat:
            labels_stat[key] = 0
    if is_all_test_labels:
        # all test labels
        chart_title = "Public Test labels Stat. Total: " + str(total_num_labels)
    elif is_correct_label:
        # correctly released label
        chart_title = "Correctly Released Student Stat" + str(total_num_labels) + "of" + str(num_answered_query)
    else:
        # released labels
        chart_title = "Released Student Stat " + str(num_answered_query) + "of" + str(total_num_queries)

    plt.title(chart_title)
    plt.xlabel('Label', weight='bold')  # fontsize=20
    plt.ylabel('Count', weight='bold')  # fontsize=20
    sns.barplot(list(labels_stat.keys()), list(labels_stat.values()))

    if is_all_test_labels:
        plt.savefig(config.save_model + "/labels_stat_" + str(total_num_labels) + ".pdf", format='pdf',
                    dpi=1200, bbox_inches="tight")
    elif is_correct_label:
        plt.savefig(config.save_model + "/correct_released_stdnt_stat_" + str(total_num_labels) + ".pdf",
                    format='pdf', dpi=1200, bbox_inches="tight")
    else:
        plt.savefig(config.save_model + "/released_stdnt_stat_" + str(num_answered_query) + ".pdf", format='pdf',
                    dpi=1200, bbox_inches="tight")
    plt.close()
    plt.show()
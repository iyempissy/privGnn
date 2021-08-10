import networkx as nx
from kmeans_pytorch import kmeans
import itertools
import random


def page_rank(public_data_edge_index, public_train_idx, public_test_idx, nb_nodes_to_select):
    # 1. create graph from public edge index and "all" public data
    # 2. Get the dictionary of the nodes and page rank value
    # 3. Select the nodes with the highest values (500 of them) that are in the public train?
    # 4. Get their keys / nodeID into an array and use it for slicing appropriately. The assumption is that the node features are arranged sequentially?

    # 1. Graph creation

    edges_raw = public_data_edge_index.cpu().numpy()
    # print("edges_raw", edges_raw)
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]

    G = nx.Graph()
    num_nodes_in_public_data = len(public_train_idx) + len(
        public_test_idx)  # total number of nodes in public data i.e both train and test #30k for reddit
    print("num_nodes_in_public_data", num_nodes_in_public_data)
    G.add_nodes_from(list(range(num_nodes_in_public_data)))
    G.add_edges_from(edges)

    # 2.
    page_rank = nx.pagerank(G, alpha=0.9)  # this is a dict
    print("page_rank", page_rank)
    print("page_len", len(page_rank))  # 30k for reddit
    # TODO we can save the page rank to file as well to shorten the run time and retrieve it later. No need it's fast

    # 3. Create a new dictionary from the old having only the data of the public train i.e use the public_train_idx to slice the page_rank dictionary

    public_train_page_rank_dict = {k.item(): page_rank[k.item()] for k in public_train_idx if
                                   k.item() in page_rank}  # check if key is in the page_rank dict and get it. Note, the keys are the nodeIDs
    print("old unsorted", public_train_page_rank_dict)

    # 3b Order them based on their values and only select 500
    public_train_page_rank_dict = dict(
        sorted(public_train_page_rank_dict.items(), key=lambda x: x[1], reverse=True)[:nb_nodes_to_select])
    print("new sorted", public_train_page_rank_dict)

    public_train_al_selection_idx = sorted(list(public_train_page_rank_dict.keys()))  # no need but let's just sort
    print("public_train_al_selection_idx", public_train_al_selection_idx)

    return public_train_al_selection_idx



def clustering(public_data_train_x, num_clusters, query_budget, device):
    # # For clustering AL
    # 1.  Cluster the nodes of G into at least k clusters e.g 10
    # 2.  To obtain initial data for training the base learner,
    # Pick N clusters2b.  select one item from each clusters and label it (Initial label set L)
    # 3.  While the labelling budget is not reached: keep selecting elements from each cluster
    # If not all cluster have say 100 elements, select the ones that have the largest and then select the remaining from there

    # Detailed algorithm but not using this. Cheated a bit by grouping into 8 clusters and selecting 125 from each to make 1K queries
    # select all keys that have a particular value from count
    # make them a new dict
    # Randomly shuffle this dict
    # select 125 from each of them (Not impelemented: if less than 100 in each cluster, select all that it has)
    # Get the keys of this inyo a list (al_selectn_idx)
    # End

    # (Not implemented) if al_selectn_idx < 1K queries (query budget),
    # Randomly select for the clusters that have above the current say 100
    # Add it into the al_selectn_idx. if that does not suffice, select a 2nd cluster n add the reamining to it

    '''pos = torch.tensor([[0., 0.], [11., 9.], [2., 8.], [2., 2.], [8., 3.], [0., 0.]])
    size = torch.Tensor([5, 5]). This needs to be defined for each dimension of the data. This can be a disadvantage cos you might as well not know the right one. Also tedious cos we have higher dimension

    cluster = grid_cluster(pos, size)
    print(cluster)  # tensor([0, 5, 3, 0, 1, 0])

    Therefore, use pytorch k-mean
    '''

    print("Cluster begins")

    # kmeans
    cluster_ids_x, cluster_centers = kmeans(
        X=public_data_train_x, num_clusters=num_clusters, distance='cosine', device=device
    )  # distance='euclidean'
    print("cluster_ids_x", cluster_ids_x)
    print("len(public_data_train_x)", len(public_data_train_x))

    cluster_ids_x = cluster_ids_x.tolist()  # convert to list

    # Stat
    max_cluster_id = max(cluster_ids_x)
    min_cluster_id = min(cluster_ids_x)
    print("max_cluster_id", max_cluster_id, "min_cluster_id", min_cluster_id)

    # count number of nodes in each cluster
    count_per_cluster = {}
    for c in range(0, num_clusters):
        count_per_cluster[c] = cluster_ids_x.count(c)

    print("count_per_cluster", count_per_cluster)
    # 10 clusters count_per_cluster {0: 129, 1: 152, 2: 725, 3: 3715, 4: 25, 5: 2426, 6: 5075, 7: 2, 8: 1944, 9: 807} ==> Euclidean
    # 10 clusters count_per_cluster {0: 1462, 1: 1002, 2: 830, 3: 1937, 4: 1248, 5: 1201, 6: 3318, 7: 911, 8: 1699, 9: 1392} ==> Cosine
    # for 8 clusters count_per_cluster {0: 127, 1: 2583, 2: 729, 3: 6078, 4: 239, 5: 1988, 6: 3241, 7: 15}
    # for 5 clusters count_per_cluster {0: 3160, 1: 3916, 2: 794, 3: 6807, 4: 323} ==> Euclidean
    # for 5 clusters count_per_cluster {0: 2571, 1: 2841, 2: 2795, 3: 5744, 4: 1049} ==> Cosine

    # Map nodes to cluster_ids
    public_train_node_idx = [i for i in range(0, len(public_data_train_x))]
    public_train_cluster_dict = {node_id: cluster for node_id, cluster in zip(public_train_node_idx, cluster_ids_x)}
    print(public_train_cluster_dict)

    # select nodes from each values (cluster ID) and insert them into list
    public_train_al_selection_idx = []
    # assume we wanna select 1K

    # query_budget = config.stdnt_share  # number of queries
    num_selected_query_per_cluster = int(query_budget / num_clusters)  # 200

    for cluster in range(0, num_clusters):
        count_in_current_key = count_per_cluster[cluster]
        if count_in_current_key > num_selected_query_per_cluster: #skip for clusters that have no much data
            all_nodes_in_cluster = [node_id for node_id, clust in public_train_cluster_dict.items() if
                                    clust == cluster]  # prints all occourence of a particular value
            # randomly select 200 from each of this all nodes cluster
            all_nodes_in_cluster = random.sample(all_nodes_in_cluster, k=num_selected_query_per_cluster)
            print("len(all_nodes_in_cluster)", len(all_nodes_in_cluster))
            public_train_al_selection_idx.append(all_nodes_in_cluster)

    # flatten the public_train_al_selection_idx
    public_train_al_selection_idx = list(itertools.chain.from_iterable(public_train_al_selection_idx))

    # randomly add numbers that are not in the public_train_al_selection_idx to make up for the not selected cluster
    # sample randomly if the clusters do not have num_selected_query_per_cluster
    while len(public_train_al_selection_idx) < query_budget:
            rnd_sample = random.choice(public_train_node_idx)
            if rnd_sample not in public_train_al_selection_idx:
                public_train_al_selection_idx.append(rnd_sample)

    print("public_train_al_selection_idx clustering", public_train_al_selection_idx)
    print("len(public_train_al_selection_idx)", len(public_train_al_selection_idx))

    return public_train_al_selection_idx
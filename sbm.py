from pysbm.test_ground import PlantedPartitionGenerator, SBMGenerator
from torch_geometric.datasets import Planetoid
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import subgraph
import matplotlib.pylab as plt
import itertools
import networkx as nx
import sys
import torch

rand_state = 100

# The reason for using SBM is to make it "similar" to the euclidean space.
# Specifically, the edges are different from the original edge of the graph. Simply, we are generating new edges for a node. As such, the connectivity is false!

# 1. Use generated SBM to train a GNN model?
# 2. Use generated graph from SBM to generate the private graphs

# Real example of generating a graph =======> This is what I should look into

from pysbm.sbm.partition import NxPartition
import numpy as np



# # =================using pyG==============
#
# dataset = Planetoid(root="./data/Cora", name="cora")
# data = dataset[0]
#
# # get the num_nodes
# num_nodes = data.num_nodes
#
# # get edge_index
# edge_index = data.edge_index
#
# labels = data.y
#
# node_features = data.x


# Old

# def create_new_private_graph(num_nodes, edge_index, labels, node_features, seed, plot_graph=False, device='cpu'):
#     """
#
#     :param num_nodes: int
#     :param edge_index: pyG COO format
#     :param labels: tensor of labels
#     :param node_features: int
#     :param seed: for seeding the graph
#     :param plot_graph: True to display or plot graph
#     :return:
#         node_features: features of the nodes for the recnostructed graph
#         reconstructed_edge_index: pyG Coo edge
#     """
#
#     # convert edge_index to (souce, target) format for networkx
#     edges_raw = edge_index.cpu().numpy()
#     # print("edges_raw", edges_raw)
#     edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
#
#     graph = nx.Graph()
#     graph.add_nodes_from(list(range(num_nodes)))
#     graph.add_edges_from(edges)
#
#     # print("Original Graph: nx.info(graph)", nx.info(graph))
#     # The y here will be a dictionary {node_id:node_label}. Like each node is assigned a label
#     all_y = labels.tolist()
#
#
#     y = {key:value for key, value in zip(range(0, num_nodes), all_y) } #partition
#     # print("cora y", y)
#     # print(max(y.values()))
#
#     if plot_graph:
#         # Original graph
#         pos = nx.spring_layout(graph)
#         # nx.draw(graph, pos)
#         nx.draw(graph, pos, node_color=[y[node] for node in graph], cmap="plasma")
#         plt.show()
#
#
#     # I think if you have your graph, just input it here directly?
#
#     # What we have:
#     # graph will be our private_graph ()
#     # y = all groundtruth labels / classes. This is the structure we are trying to preserve or the blocks in this case
#
#
#     # What we want:
#     # Build / generate new graphs from the private graph
#     # To do this, we need to find information about the edges
#     # Therefore we will use or need the edge matrix and edge probabilities
#
#     graph_information = NxPartition(graph, representation=y, fill_random=False)
#     # graph_information.get_edge_count(0,0) # gets the number of edges between 2 classes. In this case, class 0 and class 0
#
#     # B = count of classes (= 12 in this case)
#     nodes_per_block = [graph_information.get_number_of_nodes_in_block(block) for block in range(graph_information.B)]
#
#     # print("nodes_per_block", nodes_per_block)
#
#     # get_edge_count gets the number of edges between 2 classes
#     # To know how many nodes are in this category, use get_number_of_nodes_in_block
#
#     # matrix. This will give a square matrix (num_classes x num_classes)
#     edges_between_blocks = [
#         [graph_information.get_edge_count(block_left,block_right) / (
#             graph_information.get_number_of_nodes_in_block(block_left)
#             * graph_information.get_number_of_nodes_in_block(block_right)
#         )
#          for block_right in range(graph_information.B)]
#         for block_left in range(graph_information.B)]
#
#     # probability of each edge
#     edge_probabilities = np.array(edges_between_blocks)
#
#     # print("edge_probabilities.shape", edge_probabilities.shape)
#     # print("edge_probabilities", edge_probabilities)
#
#
#
#     generator = SBMGenerator(
#         number_of_blocks=graph_information.B, # number of classes
#         nodes_per_block=nodes_per_block, # list of number of nodes within each class (you want or have )
#         edge_matrix=edge_probabilities, # this is the matrix of likelihoods of creating an edge between class i,j
#         type_of_edge_matrix=SBMGenerator.PROBABILISTIC_EDGES
#     )
#
#     gen_graph, _, partition = generator.generate(False, seed=seed)
#     gen_graph_edges = gen_graph.edges()
#     # print("Final generated graph", gen_graph_edges)
#
#     pos = nx.spring_layout(graph)  # we are using the position of the original graph here
#     if plot_graph:
#         nx.draw(gen_graph, pos, node_color=[partition[node] for node in graph], cmap="plasma")
#         plt.show()
#
#
#     # Reconstruct graph from edges. Just fir fun
#     reconstruct_graph_from_edges = nx.Graph()
#     reconstruct_graph_from_edges.add_nodes_from(list(range(num_nodes)))
#     reconstruct_graph_from_edges.add_edges_from(gen_graph_edges)
#
#     # print("Graph gen from edge index nx.info(graph)", nx.info(reconstruct_graph_from_edges))
#     if plot_graph:
#         nx.draw(reconstruct_graph_from_edges, pos, node_color=[partition[node] for node in reconstruct_graph_from_edges], cmap="plasma")
#         plt.show()
#
#
#     # convert back to pyG
#     data = from_networkx(gen_graph)
#     # print("data", data)
#     reconstructed_edge_index = data.edge_index
#     # print("reconstructed_edge_index",reconstructed_edge_index)
#     # convert back to device
#     reconstructed_edge_index = reconstructed_edge_index.to(device)
#     return node_features, reconstructed_edge_index




# New
def create_new_private_graph(num_nodes, edge_index, labels, node_features, seed, create_multiple_graphs = False, num_graphs_to_create=10, plot_graph=False, device='cpu'):

    # flattened labels
    data_y_flat = torch.LongTensor(list(itertools.chain.from_iterable(labels)))
    print("data_y_flat", data_y_flat)

    # print("edge_index", edge_index.shape)
    # # create subgrah using only 266 nodes
    # idx_tensor1 = torch.ones(266, dtype=bool)
    # idx_tensor2 = torch.zeros(2442, dtype=bool)
    # all_tensor = torch.cat((idx_tensor1, idx_tensor2))
    # subgraph_g, _ = subgraph(all_tensor, edge_index)
    # print("sub", subgraph_g.shape)



    # convert edge_index to (souce, target) format for networkx
    edges_raw = edge_index.cpu().numpy()
    print("edges_raw", edges_raw)
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]

    graph = nx.Graph()
    graph.add_nodes_from(list(range(num_nodes)))
    graph.add_edges_from(edges)

    print("Original graph from edge index nx.info(graph)", nx.info(graph))

    print("Original Graph: nx.info(graph)", nx.info(graph))
    # The y here will be a dictionary {node_id:node_label}. Like each node is assigned a label
    all_y = data_y_flat.tolist()#labels.tolist()


    y = {key:value for key, value in zip(range(0, num_nodes), all_y) } #partition
    print("cora y", y)
    print(max(y.values()))

    if plot_graph:
        # Original graph
        pos = nx.spring_layout(graph)
        # nx.draw(graph, pos)
        nx.draw(graph, pos, node_color=[y[node] for node in graph], cmap="plasma")
        plt.show()


    # I think if you have your graph, just input it here directly?

    # What we have:
    # graph will be our private_graph ()
    # y = all groundtruth labels / classes. This is the structure we are trying to preserve or the blocks in this case


    # What we want:
    # Build / generate new graphs from the private graph
    # To do this, we need to find information about the edges
    # Therefore we will use or need the edge matrix and edge probabilities

    graph_information = NxPartition(graph, representation=y, fill_random=False)
    # graph_information.get_edge_count(0,0) # gets the number of edges between 2 classes. In this case, class 0 and class 0

    # B = count of classes (= 12 in this case)
    nodes_per_block = [graph_information.get_number_of_nodes_in_block(block) for block in range(graph_information.B)]

    print("nodes_per_block Original", nodes_per_block)

    if create_multiple_graphs:
        # We assume we want equal number of nodes in each graph / partition
        # Should be total_number_nodes/number of disticnt graphs we want. Say 10

        # TODO This info isnt meaningful at this moment
        nodes_per_class = int(int(num_nodes / num_graphs_to_create) / graph_information.B)
        nodes_per_block = list(np.repeat(nodes_per_class, graph_information.B)) #e.g [38,38,38...]
        print("nodes_per_block multiple graph", nodes_per_block)

    # get_edge_count gets the number of edges between 2 classes
    # To know how many nodes are in this category, use get_number_of_nodes_in_block

    # Original. We need this, so leave untouched
    # matrix. This will give a square matrix (num_classes x num_classes)
    edges_between_blocks = [
        [graph_information.get_edge_count(block_left,block_right) / (
            graph_information.get_number_of_nodes_in_block(block_left)
            * graph_information.get_number_of_nodes_in_block(block_right)
        )
         for block_right in range(graph_information.B)]
        for block_left in range(graph_information.B)]

    # probability of each edge
    edge_probabilities = np.array(edges_between_blocks)

    print("edge_probabilities.shape", edge_probabilities.shape)
    # print("edge_probabilities", edge_probabilities)


    # Generate graph
    if not create_multiple_graphs:
        # Only create one graph but with varying nodes
        generator = SBMGenerator(
            number_of_blocks=graph_information.B, # number of classes
            nodes_per_block=nodes_per_block, # list of number of nodes within each class (you want or have )
            edge_matrix=edge_probabilities, # this is the matrix of likelihoods of creating an edge between class i,j
            type_of_edge_matrix=SBMGenerator.PROBABILISTIC_EDGES
        )

        gen_graph, _, partition = generator.generate(False, seed=seed)
        gen_graph_edges = gen_graph.edges()
        # print("Final generated graph", gen_graph_edges)

        pos = nx.spring_layout(graph)  # we are using the position of the original graph here
        if plot_graph:
            nx.draw(gen_graph, pos, node_color=[partition[node] for node in graph], cmap="plasma")
            plt.show()


        # Reconstruct graph from edges. Just fir fun
        reconstruct_graph_from_edges = nx.Graph()
        reconstruct_graph_from_edges.add_nodes_from(list(range(num_nodes)))
        reconstruct_graph_from_edges.add_edges_from(gen_graph_edges)

        print("Graph gen from edge index nx.info(graph)", nx.info(reconstruct_graph_from_edges))
        if plot_graph:
            nx.draw(reconstruct_graph_from_edges, pos, node_color=[partition[node] for node in reconstruct_graph_from_edges], cmap="plasma")
            plt.show()


        # convert back to pyG
        data = from_networkx(gen_graph)
        # print("data", data)
        reconstructed_edge_index = data.edge_index
        # print("reconstructed_edge_index",reconstructed_edge_index)

        # convert back to device
        reconstructed_edge_index = reconstructed_edge_index.to(device)

        # original. already in PyG format
        return reconstructed_edge_index, node_features

    else:
         #create_multiple_graphs

        # using SBM generator based on the edges of the existing gives an edge of 82
        generator_new_graph = SBMGenerator(
            number_of_blocks = graph_information.B, # number of classes
            nodes_per_block = nodes_per_block, # We changed this above for multiple classes
            edge_matrix = edge_probabilities,
            type_of_edge_matrix = SBMGenerator.PROBABILISTIC_EDGES
        )

        #  # using planted parition generator, we can somewhat control the edges
        # generator_new_graph = PlantedPartitionGenerator(
        #      number_of_groups=graph_information.B, # number of classes
        #      number_of_vertices_in_each_group=nodes_per_class, # multiply this by graph_information.B to get the total number of nodes in each graph
        #      edge_probability_in_group= 0.1, #.2,
        #      edge_probability_between_groups=0.01  # .01
        #  )


        all_gen_graph = []

        # recreate node node_features
        # corresponding = corr
        all_corr_nodes_features = np.array_split(node_features.cpu(), num_graphs_to_create) #array of tnesors for the corresponding nodes
        all_corr_nodes_labels = np.array_split(labels.cpu(), num_graphs_to_create)

        # print("all_corr_nodes", all_corr_nodes_features)
        # print("all_corr_nodes_labels", all_corr_nodes_labels)

        total_nodes_per_gen_graph = nodes_per_class * graph_information.B

        # recreate graphs
        # convert each to torch but still keep them in the list
        for i in range(num_graphs_to_create):
            print("i", i)
            each_gen_graph, _, partitions = generator_new_graph.generate(False, seed=seed+i)
            each_gen_graph_edges = each_gen_graph.edges()



            # Quick plot
            # Reconstruct graph from edges. Just fir fun
            reconstruct_graph_from_edges = nx.Graph()
            reconstruct_graph_from_edges.add_nodes_from(list(range(total_nodes_per_gen_graph)))
            reconstruct_graph_from_edges.add_edges_from(each_gen_graph_edges)

            print("Small Graph gen from edge index nx.info(graph) for graph", i,  nx.info(reconstruct_graph_from_edges))
            if plot_graph:
                nx.draw(reconstruct_graph_from_edges, pos,
                        node_color=[partitions[node] for node in reconstruct_graph_from_edges], cmap="plasma")
                plt.show()





            # convert back to pyG
            data = from_networkx(each_gen_graph)
            # print("data", data)
            reconstructed_edge_index = data.edge_index
            # print("reconstructed_edge_index",reconstructed_edge_index)

            # convert back to device
            reconstructed_edge_index = reconstructed_edge_index.to(device)



            all_gen_graph.append(reconstructed_edge_index)

            # print("Original all_corr_nodes_features[0].shape", all_corr_nodes_features[i].shape)
            # Modify to the nodes only select say a certain number of nodes and labels from it to fit into the nodes in each graph that you want
            all_corr_nodes_features[i] = all_corr_nodes_features[i][:total_nodes_per_gen_graph]
            all_corr_nodes_labels[i] = all_corr_nodes_labels[i][:total_nodes_per_gen_graph]
            # print("Modified all_corr_nodes_features[0].shape", all_corr_nodes_features[i].shape)



        # Converted to PyG format. whenever I wanna use
        # this returns a list. So need to access each and then convert
        # edge_idx0 = all_gen_graph[0] where 0 is the graph I wanna use. The max is num_graphs_to_create.
        # The same for nodes but for this, we only access by nodes_features0 = all_corr_nodes_features[0]

        return all_gen_graph, all_corr_nodes_features, all_corr_nodes_labels






# create_new_private_graph(num_nodes, edge_index, labels, node_features, seed=rand_state, plot_graph=False)
# for i in range(43, 50):
#     graph, _, partition = generator.generate(False, seed=i)
#     nx.draw(graph, pos, node_color=[partition[node] for node in graph], cmap="plasma")
#     plt.show()

# Is the generated graph only for 1 class or for different classes?
# Yes it's for different classes. Inspect the original input graph in terms of number of nodes vs the generated one. They are the same

# I will have to do this for the private graph to get the "new graph". Then use the generator to generate multiple instances of the same graph

# nodes_per_block is fixed to the total number of nodes that we have in each class in the original graph. But we can freely adjust it e.g making less nodes
# Our objective, we have the graph and the parition and we want tot get new instances. Lokking similar but the edges are not the same as the one in the original graph
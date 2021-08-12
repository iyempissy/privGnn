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

# New
def create_new_private_graph(num_nodes, edge_index, labels, node_features, seed, create_multiple_graphs = False, num_graphs_to_create=10, plot_graph=False, device='cpu'):

    # flattened labels
    data_y_flat = torch.LongTensor(list(itertools.chain.from_iterable(labels)))
    print("data_y_flat", data_y_flat)

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


    graph_information = NxPartition(graph, representation=y, fill_random=False)

    # B = count of classes (= 12 in this case)
    nodes_per_block = [graph_information.get_number_of_nodes_in_block(block) for block in range(graph_information.B)]

    print("nodes_per_block Original", nodes_per_block)

    if create_multiple_graphs:
        nodes_per_class = int(int(num_nodes / num_graphs_to_create) / graph_information.B)
        nodes_per_block = list(np.repeat(nodes_per_class, graph_information.B)) #e.g [38,38,38...]
        print("nodes_per_block multiple graph", nodes_per_block)

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

        pos = nx.spring_layout(graph)  # we are using the position of the original graph here
        if plot_graph:
            nx.draw(gen_graph, pos, node_color=[partition[node] for node in graph], cmap="plasma")
            plt.show()


        reconstruct_graph_from_edges = nx.Graph()
        reconstruct_graph_from_edges.add_nodes_from(list(range(num_nodes)))
        reconstruct_graph_from_edges.add_edges_from(gen_graph_edges)

        print("Graph gen from edge index nx.info(graph)", nx.info(reconstruct_graph_from_edges))
        if plot_graph:
            nx.draw(reconstruct_graph_from_edges, pos, node_color=[partition[node] for node in reconstruct_graph_from_edges], cmap="plasma")
            plt.show()


        # convert back to pyG
        data = from_networkx(gen_graph)
        reconstructed_edge_index = data.edge_index

        # convert back to device
        reconstructed_edge_index = reconstructed_edge_index.to(device)

        # original. already in PyG format
        return reconstructed_edge_index, node_features

    else:

        # using SBM generator based on the edges of the existing gives an edge of 82
        generator_new_graph = SBMGenerator(
            number_of_blocks = graph_information.B, # number of classes
            nodes_per_block = nodes_per_block, # We changed this above for multiple classes
            edge_matrix = edge_probabilities,
            type_of_edge_matrix = SBMGenerator.PROBABILISTIC_EDGES
        )

        all_gen_graph = []

        # recreate node node_features
        all_corr_nodes_features = np.array_split(node_features.cpu(), num_graphs_to_create) #array of tnesors for the corresponding nodes
        all_corr_nodes_labels = np.array_split(labels.cpu(), num_graphs_to_create)


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
            reconstructed_edge_index = data.edge_index

            # convert back to device
            reconstructed_edge_index = reconstructed_edge_index.to(device)



            all_gen_graph.append(reconstructed_edge_index)

            # Modify to the nodes only select say a certain number of nodes and labels from it to fit into the nodes in each graph that you want
            all_corr_nodes_features[i] = all_corr_nodes_features[i][:total_nodes_per_gen_graph]
            all_corr_nodes_labels[i] = all_corr_nodes_labels[i][:total_nodes_per_gen_graph]


        # Converted to PyG format. whenever I wanna use
        # this returns a list. So need to access each and then convert
        # edge_idx0 = all_gen_graph[0] where 0 is the graph I wanna use. The max is num_graphs_to_create.
        # The same for nodes but for this, we only access by nodes_features0 = all_corr_nodes_features[0]

        return all_gen_graph, all_corr_nodes_features, all_corr_nodes_labels
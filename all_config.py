from easydict import EasyDict as edict

config = edict()
# Compute baselines
config.compute_baselines = True # if false, no baseline will be computed.

# ensure clippung is false to avoid error
config.use_pate_graph = False # The goal is to use the multiple graphs created by the SBM or randomsplit for training different nodes
config.split_graph_randomly = False # if true, then it doesn't use SBM and creates only multiple subgraph. config.use_pate_graph needs to be true also. Set to false to use SBM created multiple graphs
config.use_mlp = False # the mlp is used instead of graph for the private data. Only private features is used. Thus, set split_graph_randomly to True for faster result
config.num_teacher_graphs = 10

# method TKT
config.is_tkt = False#True
config.use_sbm = False # use SBM approach i.e generate a synthetiv links between graphs. This also affects the baseline1_star.


# Dataset
config.is_reddit_dataset = False#True
config.is_amazon_dataset = False
config.is_cora_dataset = False
config.is_arxiv_random_split = True
# If all of these is False, then it's Arxiv

if config.is_reddit_dataset:
    config.nb_labels = 41  # hardcoded for reddit dataset
    config.nb_features = 602
    config.data_name = "Reddit"
    # nb_teachers is the number of private nodes to select
    config.nb_teachers = 1000
    config.delta = 10e-5
elif config.is_amazon_dataset:
    config.nb_labels = 10  # hardcoded for amazon dataset, computers = 10, photo=8
    config.nb_features = 767 #computers = 767, photo=745
    config.data_name = "Amazon"
    config.nb_teachers = 750
    config.delta = 10e-4
elif config.is_cora_dataset:
    config.nb_labels = 7  # hardcoded for cora dataset
    config.nb_features = 1433
    config.data_name = "Cora"
    config.nb_teachers = 300
    config.delta = 10e-4
elif config.is_arxiv_random_split:
    config.nb_labels = 40  # hardcoded for arxiv dataset
    config.nb_features = 128
    config.data_name = "ArxivRandom"
    config.nb_teachers = 300
    config.delta = 10e-5

else:
    config.nb_labels = 40  # hardcoded for arxiv dataset
    config.nb_features = 128
    config.data_name = "Arxiv"
    config.nb_teachers = 1000
    config.delta = 10e-5  # 0.00001

config.confident = True
config.sigma1 = 75

config.threshold = 10
config.gau_scale = 25

# if this is a confidence based methods, sigma1 is used for selection, and gau_scale is added to voting

args_log_steps = 1
config.num_layers = 4  # 3
config.hidden_channels = 64 #256
config.dropout = 0.5
config.num_runs = 11  # 1 # number of times to run experiment

config.gpu_devices = '1'


config.use_sage = True
config.extract_features = "normal"  # feature #normal

config.data_dependent_rdp = False

# These are for cliping logits
config.is_clip_logit = False #True
config.add_gaussian_noise_logit = False #for adding noise to logit

config.prob = 0.30 #0.15  # The sub-sampling ratio i.e Gamma To usee all data set to 1
config.stdnt_share = config.nb_nodes_to_select = 1000 #1000  # num of queries from students

# For adding noise directly to posteriors.
config.epsilon = 0.8
config.use_lap_noise = True # it will use gaussian noise with the delta if set to False

# config.stepsize =20
config.student_epoch = 500

config.save_model = "save_model/graph"+config.data_name

config.lr = 0.01

# These are active learning approach are all depenedent on TKT
config.use_al = False
config.use_pagerank = False#True # page rank centrality
config.use_clustering = False # clustering approach
config.num_clusters = 10  # 10


if config.compute_baselines:
    config.log_filename = "stdoutnew1Base"+str(config.epsilon)+str(config.nb_teachers)+str(config.stdnt_share)+config.data_name+"Baseline.txt" # filename for saving all print statement
elif config.is_tkt:
	# TKT
	config.log_filename = "stdoutnew1TKT"+str(config.epsilon)+str(config.nb_teachers)+str(config.stdnt_share)+".txt"+config.data_name # filename for saving all print statement
else:
    # this is PATE
    if config.use_mlp:
        ispateMLP = "withMLP"
    else:
        ispateMLP = "withoutMLP"

    config.log_filename = "stdoutnew1PATE" + ispateMLP + str(
        config.epsilon) +str(config.stdnt_share)+ ".txt" + config.data_name  # filename for saving all print statement
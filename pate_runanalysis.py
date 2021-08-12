from syft.frameworks.torch.dp import pate as pate_analysis
import numpy as np
import sys

import all_config as config
config = config.config





# Only to be run on my system! Problem with running on server

data_type = "Arxiv"  # Arxiv, Reddit, Amazon
epsilon = 1

if data_type == "Reddit":
    delta = 10e-5  # 0.00008
elif data_type == "Arxiv":
    delta = 10e-5  # 0.00001
elif data_type == "Amazon":
    delta = 10e-4  # 0.0004


withMLP = True
if withMLP:
    pate_type = "PATEwithMLP"
else:
    pate_type = "PATEwithoutMLP"


# load array
# 500 queries
non_noisy_preds = np.load(
    "stdoutnew1" + pate_type + str(epsilon) + "500.txt" + data_type + str(epsilon) + "teacherpred.npy")
stdnt_labels = np.load("stdoutnew1" + pate_type + str(epsilon) + "500.txt" + data_type + str(epsilon) + "stdntlabels.npy")

# 1K queries
# non_noisy_preds = np.load(
#     "stdoutnew1" + pate_type + str(epsilon) + ".txt" + data_type + str(epsilon) + "teacherpred.npy")
# stdnt_labels = np.load("stdoutnew1" + pate_type + str(epsilon) + ".txt" + data_type + str(epsilon) + "stdntlabels.npy")

print("non_noisy_preds", non_noisy_preds)
print("stdnt_labels", stdnt_labels)

data_dep_eps, data_ind_eps = pate_analysis.perform_analysis(teacher_preds=non_noisy_preds, indices=stdnt_labels,
                                                            noise_eps=epsilon, delta=delta, moments=8)
print("Data Independent Epsilon:", data_ind_eps)
print("Data Dependent Epsilon:", data_dep_eps)


sys.exit()

# withMLP = True
for withMLP in [True, False]:
    for epsilon in [0.1, 0.2, 0.4, 0.8, 1]:
        # epsilon = 0.1
        for data_type in ["Reddit", "Amazon"]:#, "Arxiv"]:
            # data_type = "Cora"  # Reddit, Amazon, Arxiv
            if withMLP:
                pate_type = "PATEwithMLP"
            else:
                pate_type = "PATEwithoutMLP"


            if data_type == "Reddit":
                delta = 0.00008
            elif data_type == "Arxiv":
                delta = 0.00001
            elif data_type == "Amazon":
                delta = 0.0004

            print("stdoutnew1"+pate_type+str(epsilon)+".txt"+data_type+str(epsilon)+"teacherpred.npy")
            # print("stdoutnew1"+pate_type+str(epsilon)+".txt"+data_type+str(epsilon)+"stdntlabels.npy")

            # load array
            non_noisy_preds = np.load("stdoutnew1"+pate_type+str(epsilon)+".txt"+data_type+str(epsilon)+"teacherpred.npy")
            stdnt_labels = np.load("stdoutnew1"+pate_type+str(epsilon)+".txt"+data_type+str(epsilon)+"stdntlabels.npy")

            data_dep_eps, data_ind_eps = pate_analysis.perform_analysis(teacher_preds=non_noisy_preds, indices=stdnt_labels,
                                                               noise_eps=epsilon, delta=delta)
            print("Data Independent Epsilon:", data_ind_eps)
            print("Data Dependent Epsilon:", data_dep_eps)
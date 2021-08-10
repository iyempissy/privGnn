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

# Not working: TODO: Save, then run later!
data_dep_eps, data_ind_eps = pate_analysis.perform_analysis(teacher_preds=non_noisy_preds, indices=stdnt_labels,
                                                            noise_eps=epsilon, delta=delta, moments=8)
print("Data Independent Epsilon:", data_ind_eps)
print("Data Dependent Epsilon:", data_dep_eps)


sys.exit()








# TODO: change delta
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



            # Not working: TODO: Save, then run later!
            data_dep_eps, data_ind_eps = pate_analysis.perform_analysis(teacher_preds=non_noisy_preds, indices=stdnt_labels,
                                                               noise_eps=epsilon, delta=delta)
            print("Data Independent Epsilon:", data_ind_eps)
            print("Data Dependent Epsilon:", data_dep_eps)



with sns.plotting_context("notebook", font_scale=1.35, rc={"axes.linewidth":2, "axes.labelsize":20}):
  g = sns.FacetGrid(amaarxred_privutil_df, col="Data", hue="Type")
  g.map(sns.lineplot, "Epsilon", "Accuracy", data = amaarxred_privutil_df,  hue="Type", linewidth = 3, style="Type", dashes=False, markers=["X","s", "^", "o", "D"], markersize=10)
  g.add_legend(title="") #Display legend and remove legend title

plt.subplots_adjust(top=0.9)
#g.fig.suptitle("Joint Homophily Distribution in First Explanation");

[plt.setp(ax.yaxis.get_majorticklabels(), fontsize=15, weight="bold") for ax in g.axes.flat];
[plt.setp(ax.xaxis.get_majorticklabels(), fontsize=15, weight="bold") for ax in g.axes.flat];

#[ax.xaxis.label.set_size(18) for ax in g.axes.flat];
# x and y label
[ax.set_ylabel("Accuracy", weight="bold") for ax in g.axes.flat[:1]];
[ax.set_xlabel(r"$\mathbf{\epsilon_{noise}}$", weight="bold") for ax in g.axes.flat];

# font size
[ax.yaxis.label.set_size(18) for ax in g.axes.flat];
[ax.xaxis.label.set_size(20) for ax in g.axes.flat];
[ax.title.set_size(20) for ax in g.axes.flat];

# set x axis values
g.set(xticks=[0.2, 0.4, 0.6, 0.8, 1])

# # change legend text size
# # plt.setp(g._legend.get_texts(), fontsize=20)

# # setting xlabel for each of the chart
# g.axes[0,0].set_xlabel("Noise Epsilon", weight="bold") #fontsize=20 already done above in plotting_context
# g.axes[0,1].set_xlabel("Noise Epsilon", weight="bold")
# g.axes[0,2].set_xlabel("Noise Epsilon", weight="bold")

# # set ylabel
# g.axes[0,0].set_ylabel("Accuracy",  weight="bold")

plt.savefig("priv-utilamaarxred.pdf", format='pdf', dpi=1200, bbox_inches="tight")

plt.show()
# sys.exit()
from autodp import rdp_bank, dp_acct, rdp_acct, privacy_calibrator

import torch

noise = torch.normal(mean=0.0, std=0.2, size=(10, 10))

print(noise)

import all_config as config
config = config.config

acct = rdp_acct.anaRDPacct()
acct_2 = rdp_acct.anaRDPacct()
dependent_acct = rdp_acct.anaRDPacct()
delta = config.delta #0.0001#config.delta i.e 1/10,000
sigma = config.sigma1  # gaussian parameter
gau_scale = config.gau_scale
print(sigma)
print("gauscale", gau_scale)
gaussian = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x) #sensitivity is 1 here
gaussian2 = lambda x: rdp_bank.RDP_inde_pate_gaussian({'sigma': gau_scale}, x) #config.sigma1 this should be used cos sensitivity =2

print("gaussian2", gaussian2)

acct.compose_poisson_subsampled_mechanisms(gaussian, 0.15, 10000)  # 1K is the same as coeff=len(teachers_preds)i.e num_queries len(teachers_preds) is the total amount of queries
acct_2.compose_poisson_subsampled_mechanisms(gaussian2, 0.15, 735)  # 1K is the same as coeff=len(teachers_preds)i.e num_queries len(teachers_preds) is the total amount of queries



# should only be 1 in the format string
noisy_screening_comp = acct.get_eps(delta) #===>This is the RDP epsilon {lower bound for the subsampling poisson}
noisy_aggr = acct_2.get_eps(delta)
print("Composition of student Gaussian mechanisms gives {} ".format(noisy_screening_comp))
print("Composition of aggr Gaussian mechanisms gives {} ".format(noisy_aggr))

# Laplace

# data_type = "Amazon"  # Reddit, Amazon,
for data_type in ["Amazon", "Reddit", "Arxiv"]:
    for epsilon in [0.1, 0.2, 0.4, 0.8, 1]:
        # epsilon = 0.8
        for num_queries in [1000, 500]:
            print("num_queries", num_queries, "data_type", data_type, "epsilon", epsilon)
            # num_queries = 1000
            sampling_ratio = 0.1


            if data_type == "Reddit":
                delta = 10e-5 #0.00008
            elif data_type == "Arxiv":
                delta = 10e-5 #0.00001
            elif data_type == "Amazon":
                delta = 10e-4 #0.0004

            acct = rdp_acct.anaRDPacct()
            dependent_acct = rdp_acct.anaRDPacct()


            # delta = 0.0001#config.delta
            sigma = config.sigma1  # gaussian parameter
            b = 1/epsilon #config.epsilon #2.5
            laplacian = lambda x: rdp_bank.RDP_laplace({'b': b}, x)

            print("laplacian", laplacian)

            acct.compose_poisson_subsampled_mechanisms(laplacian, sampling_ratio, num_queries)  # 1K is the same as coeff=len(teachers_preds)i.e num_queries len(teachers_preds) is the total amount of queries

            e_dp_epsilon_comp = acct.get_eps(delta) #===>This is the epsilon delta dp
            print("Composition of \epsilon, delta Laplacian mechanisms gives {} epsilon".format(e_dp_epsilon_comp))


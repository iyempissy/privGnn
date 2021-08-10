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
# gaussian2 = lambda x: rdp_bank.RDP_inde_pate_gaussian({'sigma': 200}, x) #config.sigma1 this should be used cos sensitivity =2

print("gaussian2", gaussian2)
# TODO gaussian2 is the normal RDP computatyion of Gaussian mechanism. Doesnt take into account the gamma or subsampling

# TODO Moment accountant for gaussian i.e noisy screening ==> This is the lower bound for the poison subsampling
acct.compose_poisson_subsampled_mechanisms(gaussian, 0.15, 10000)  # 1K is the same as coeff=len(teachers_preds)i.e num_queries len(teachers_preds) is the total amount of queries
acct_2.compose_poisson_subsampled_mechanisms(gaussian2, 0.15, 735)  # 1K is the same as coeff=len(teachers_preds)i.e num_queries len(teachers_preds) is the total amount of queries

# TODO: How to know which alpha is selected?


# print("Composition of student   Gaussian mechanisms gives {} ".format(acct.get_eps(delta), delta))
# should only be 1 in the format string
noisy_screening_comp = acct.get_eps(delta) #===>This is the RDP epsilon {lower bound for the subsampling poisson}
noisy_aggr = acct_2.get_eps(delta)
print("Composition of student Gaussian mechanisms gives {} ".format(noisy_screening_comp))
print("Composition of aggr Gaussian mechanisms gives {} ".format(noisy_aggr))

# for Gaussian
# Larger std = wider = more noise
# e.g noise of 0.05 will give b=20 when sensitivity =1

# when sesnitivity = 2. Delta= 1e-5.
# sigma: final RDP epsilon
# 5 = 17.157
# 25 = 2.729 gau_scale
# 75 = 0.875 sigma1
# 100 = 0.653
# 200 = 0.324

# TODO Actually, no need, just plug in the value. Thats all!
#  1. Need to reimplemet our own "Gaussian2" according to our 12m etc.
#  2. We need to reimplement our tightbound
#  Then we can convert it to E delta by using the "tight" bound of our mechanism




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
            laplacian = lambda x: rdp_bank.RDP_laplace({'b': b}, x) #config.sigma1 this should be used cos sensitivity =2

            print("laplacian", laplacian)
            # TODO gaussian2 is the normal RDP computatyion of Gaussian mechanism. Doesnt take into account the gamma or subsampling

            # TODO Moment accountant for gaussian i.e noisy screening ==> This is the lower bound for the poison subsampling
            acct.compose_poisson_subsampled_mechanisms(laplacian, sampling_ratio, num_queries)  # 1K is the same as coeff=len(teachers_preds)i.e num_queries len(teachers_preds) is the total amount of queries

            # TODO: How to know which alpha is selected?


            # print("Composition of student   Gaussian mechanisms gives {} ".format(acct.get_eps(delta), delta))
            # should only be 1 in the format string
            e_dp_epsilon_comp = acct.get_eps(delta) #===>This is the epsilon delta dp
            print("Composition of \epsilon, delta Laplacian mechanisms gives {} epsilon".format(e_dp_epsilon_comp))


# for laplacian
# smaller gamma = wider = more noise
# e.g noise (gamma) of 0.05 will give b=20 when sensitivity =1

# Delta= 1e-5.
# beta: final RDP epsilon
# 2.5: 24.418641528912477 \gamma has to be 0.4 if senstitivity = 1 and 0.8 if sensitivity =2
# 5: 10.613807515479404
# 10: 4.914761832245278
# 15: 3.196593817199753
# 20:  2.365899983014982 \gamma has to be 0.05 if senstitivity = 1 and 0.1 if sensitivity =2
# 50: 0.9239430386891143
# 75: 0.6125272317654111
#100: 0.4581322829463148  \gamma has to be 0.01 if senstitivity = 1 and 0.02 if sensitivity =2


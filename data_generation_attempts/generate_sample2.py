import numpy as np
import matplotlib.pyplot as plt


def likelihood(d, p): 
    l = 1
    for i in d:
        if i == 1:
            l = l*p
        else:
            l = l * (1-p)
    
    return l


    # ret = 0
    # s = sum(d)
    # # print(s)
    # # print(len(d) - s)
    # ret += np.power(p, s)
    # ret *= np.power((1-p), len(d) - s)
    # return ret

def specific_vector(d):
    def call(p):
        return likelihood(d, p)
    return call

quarter = [1,1,1,1,0,1,1,0,0,1]
penny = [1,0,0,0,0,0,0,1,0,0]
all_flips = quarter + penny
print(all_flips)
print(likelihood(quarter, 0.5))
x = np.arange(0, 1, 0.05)
quarter_likelihoods = list(map(specific_vector(quarter), x))
penny_likelihoods = list(map(specific_vector(penny), x))

print(x)
# print(likelihoods)
# plt.hist(x, bins=x, weights=all_likelihoods)
# plt.show()

all_likelihoods_matrix = np.outer(quarter_likelihoods, penny_likelihoods)
z_data = all_likelihoods_matrix.flatten()


#################### - 3d Histogram
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.bar3d( quarter_likelihoods,
#           penny_likelihoods,
#           np.zeros(len(z_data)),
#           np.ones(10), np.ones(10), z_data )
# plt.show()
#################### - 3d END



#################### - LLR Test
def flipcoin(p, n): return np.random.choice([0,1], replace=True, p=(p, 1-p), size=n)
    

# heads_probability = 0.6
# flips_per_experiment = 100
# flips = flipcoin(heads_probability, flips_per_experiment)
# print(flips)

def calc_max_likelihood_params (flips, number_params):
    if number_params > 1: 
        splits = np.array_split(flips, number_params)
    else: 
        splits = [flips]
    probs = []
    for s in splits:
        probs.append(sum(s)/len(s))
    
    return probs

def calculate_likelihood(flips, param_values):
    if len(param_values) > 1: 
        splits = np.array_split(flips, len(param_values))
    else:
        splits = [flips]

    likelihoods = []
    for index, s in enumerate(splits):
        l = likelihood(s, param_values[index])
        likelihoods.append(l)
    
    return np.product(likelihoods)

def calculate_likelihood_ratio(d, ps1, ps2):
    l1 = calculate_likelihood(d, ps1)
    l2 = calculate_likelihood(d, ps2)

    lr = l1/l2

    lrt = -2 * np.log(lr)

    return lrt


def calc_lrts(flips, num_params_model_1, num_params_model_2):
    lrts = []
    for i in range(len(flips)):
        f = flips[i]
        max_probs_model_1 = calc_max_likelihood_params(f, num_params_model_1)
        max_probs_model_2 = calc_max_likelihood_params(f, num_params_model_2)

        statistic = calculate_likelihood_ratio(f, max_probs_model_1, max_probs_model_2)

        lrts.append(statistic)

    return lrts



flips_per_experiment = 1000
# flips_per_experiment = 4000
covariates = 0.2 * flips_per_experiment
n_experiments = 2000
flips = []
prob_heads = 0.8
for i in range(n_experiments):
     flips.append(list((flipcoin(prob_heads,flips_per_experiment))))

    
LRTs = calc_lrts(flips,1, 6)
# LRTs = calc_lrts(flips, covariates - 1, covariates)
print(LRTs)

fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(LRTs, bins=50)
plt.show()

# print(calc_max_likelihood_params ([1,1,0,1], 2))
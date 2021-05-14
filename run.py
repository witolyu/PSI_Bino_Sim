import scipy.special
from scipy.stats import binom
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time
import sys
import configparser

start_time = time.time()

def print_dist(dist):
    m = len(dist)
    print("r\tp(r)")
    for i in range(m):
        print(str(i) + "\t" + str(dist[i]))


def return_mean_and_variance(dist):
    mean = 0.0
    variance = 0.0
    length = len(dist)
    for idx in range(length):
        mean += idx * dist[idx]

    for idx in range(length):
        variance += idx ** 2 * dist[idx]
    variance -= mean ** 2
    return mean, variance


# setting the values
def total_variation(dist1, dist2):
    ss = min(len(dist1), len(dist2))
    result = 0.0
    for i in range(ss):
        result += abs(dist1[i] - dist2[i])

    if len(dist1) > ss:
        for i in range(ss, len(dist1)):
            result += dist1[i]
    else:
        for i in range(ss, len(dist2)):
            result += dist2[i]
    return result / 2


# calculate mixture distribution
def mix_distribution(dist_list, prob_weight_list):
    support = len(dist_list[0])
    length = len(dist_list)
    mixed_dist = [0] * support
    for idx in range(length):
        # calculate normalized "distribution"
        ls = [x * prob_weight_list[idx] for x in dist_list[idx]]
        mixed_dist = [a + b for a, b in zip(mixed_dist, ls)]
    return mixed_dist


# this function return a list representing the number of P1's input sets of size 0, 1,..., n. (Fix P2's input set)
def num_inputs_with_intersection_size(s, n):
    result = []
    for z in range(n + 1):
        result.append(scipy.special.comb(n, z, exact=True) * scipy.special.comb(s - n, n - z, exact=True))
    # max_value = max(result)
    # max_index = result.index(max_value)
    # print(max_index)
    return result


# return a list of m permutations.
def generate_permutations(d, m):
    perms = []
    for idx in range(m):
        temp = list(range(d))
        random.shuffle(temp)
        perms.append(temp)
    return perms


# return the output distribution from the ideal world, given a particular input.
def ideal_world_dist(p1, p2, n, m, p_set):
    assert len(p1) == len(p2) == n
    # receive the real intersection size.
    intersection_size = len(list(set(p1) & set(p2)))
    # print("intersection size: ", intersection_size)

    # probability that applying each hash function results in a matching min values, i.e., intersection size/union size.
    # Not counting collision of non-matching values.
    p_w = intersection_size / (2 * n - intersection_size)

    # Each success has 1 - p_set to be reverted, so the mixture distribution correponds to a binomial dist with success
    # probabiltiy p_w * p_set
    dist = [binom.pmf(r, m, p_w * p_set) for r in range(m + 1)]

    return dist


# return the output distribution from the real world, given a particular input and a set of fixed hash functions.
def real_world_dist(p1, p2, n, m, p_set, hash_functions):
    assert len(p1) == len(p2) == n
    assert len(hash_functions) == m
    eqa_vec_weight = 0
    for hash_function in hash_functions:
        p1_min = min([hash_function[p1[idx]] for idx in range(n)])
        p2_min = min([hash_function[p2[idx]] for idx in range(n)])
        eqa_vec_weight += (p1_min == p2_min)
    dist = [binom.pmf(r, eqa_vec_weight, p_set) for r in range(m + 1)]
    return dist


def pairwise_input_experiment(d, n, m, p_set):
    # given a pair of inputs for P2, examine both output distributions in both real and ideal worlds.
    # Input of party 1.
    domain_set = list(range(d))  # 0,1,...,domain_size-1
    p1 = list(range(n))  # 0,1,...,input_size-1

    # random.seed(10)
    p2_A = random.sample(domain_set, n)  # Sample without replacement
    p2_B = random.sample(domain_set, n)  # Sample without replacement

    print("P1's input: ", p1)
    print("First P2's input: ", p2_A)
    print("Second P2's input: ", p2_B)

    dist_A_ideal = ideal_world_dist(p1, p2_A, n, m, p_set)
    dist_B_ideal = ideal_world_dist(p1, p2_B, n, m, p_set)

    # random.seed(100)
    hash_functions = generate_permutations(d, m)
    dist_A_real = real_world_dist(p1, p2_A, n, m, p_set, hash_functions)
    dist_B_real = real_world_dist(p1, p2_B, n, m, p_set, hash_functions)

    tv_ideal = total_variation(dist_A_ideal, dist_B_ideal)
    tv_real = total_variation(dist_A_real, dist_B_real)
    print("The total variation between P2's two inputs in the ideal world is: ", tv_ideal)
    print("The total variation between P2's two inputs in the real world is: ", tv_real)
    return 0


def uniformly_sampled_input_experiment(d, n, m, p_set, sample_size):
    # sample inputs
    domain_set = list(range(d))  # 0,1,...,domain_size-1
    p1 = list(range(n))  # 0,1,...,input_size-1

    # random.seed(10)
    # being lazy below, rather than uniformly chosen, input is sampled from the domain of all inputs with replacement.

    p2_list = []
    for idx in range(sample_size):
        p2_list.append(random.sample(domain_set, n))

    print("P1's input: ", p1)
    print("First P2's input: ", p2_list[0])

    output_dist_list_real = []
    output_dist_list_ideal = []

    # random.seed(100)
    hash_functions = generate_permutations(d, m)

    # uniform probability
    weight_list = [1 / sample_size] * sample_size

    for idx in range(sample_size):
        output_dist_list_real.append(real_world_dist(p1, p2_list[idx], n, m, p_set, hash_functions))
        output_dist_list_ideal.append(ideal_world_dist(p1, p2_list[idx], n, m, p_set))

    print("P1's input: ", p1)
    print("First P2's input: ", p2_list[0])

    mix_real = mix_distribution(output_dist_list_real, weight_list)
    mix_real_mean, mix_real_var = return_mean_and_variance(mix_real)
    mix_ideal = mix_distribution(output_dist_list_ideal, weight_list)
    mix_ideal_mean, mix_ideal_var = return_mean_and_variance(mix_ideal)

    tv = total_variation(mix_real, mix_ideal)
    print("The total variation between the mixture distribution of the sampled inputs between the two worlds is"
          , tv)

    print("--- %s seconds ---" % (time.time() - start_time))

    plt.figure(figsize=(10,10))

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_ylim(0, 1.1 * max(mix_real + mix_ideal))
    plt.bar(list(range(m + 1)), mix_real)
    plt.title('Real world, mean=%f, variance=%f' % (mix_real_mean, mix_real_var))

    ax2 = plt.subplot(1, 2, 2, sharey=ax1)
    plt.bar(list(range(m + 1)), mix_ideal)
    plt.title('Ideal world, mean=%f, variance=%f' % (mix_ideal_mean, mix_ideal_var))

    plt.suptitle('Domain_size: %i, Input_size: %i, Input_domain_size: %s,\n sample size: %i, # hashes functions: %i, '
                 ' Total variation: %f'
                 % (d, n, "{:.2e}".format(scipy.special.comb(d, n, exact=False)), sample_size, m, tv))
    plt.tight_layout()
    plt.savefig('experiment.png', dpi=100, bbox_inches="tight")
    plt.show()
    return 0


# print_dist(dist_c)

# plt.bar(r_values, dist_c)
# plt.show()

# ---- Real world distribution ----
# Before seeing hash function. Assume uniformity on input set.
# num_inputs_with_intersection_size(10000, 500)

# Pr[z = 0], ... Pr[z = input_size]


# Pr[w = 0], ... Pr[w = m]

def main():
    # Domain size
    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    domain_size = int(config['parameters']['domain_size'])
    input_size = int(config['parameters']['input_size'])
    m = int(config['parameters']['hash_number'])
    p_set = float(config['parameters']['p_set'])
    sample_size = int(config['parameters']['sample_size'])

    # Input size
    # input_size = 50

    # number of hash function
    # m = 10

    # Probability to set each bit of binary mask to 1.
    # p_set = 0.5

    assert domain_size >= input_size

    # random.seed(100)

    # print("\n==========Pairwise input experiment==========")
    # pairwise_input_experiment(domain_size, input_size, m, p_set)

    print("\n==========Uniformly sampled input experiment==========")
    # sample size
    # sample_size = 10000
    # print("domain size = ", scipy.special.comb(domain_size, input_size, exact=False))

    uniformly_sampled_input_experiment(domain_size, input_size, m, p_set, sample_size)



# Using the special variable
# __name__
if __name__ == "__main__":
    main()

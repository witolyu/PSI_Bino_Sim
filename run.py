import scipy.special
from scipy.stats import binom
import matplotlib.pyplot as plt


def print_dist(dist):
    m = len(dist)
    print("r\tp(r)")
    for i in range(m):
        print(str(i) + "\t" + str(dist[i]))


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


# this function return a list representing the number of P1's input sets of size 0, 1,..., n. (Fix P2's input set)
def num_inputs_with_intersection_size(s, n):
    result = []
    for z in range(n + 1):
        result.append(scipy.special.comb(n, z, exact=True) * scipy.special.comb(s - n, n - z, exact=True))
    # max_value = max(result)
    # max_index = result.index(max_value)
    # print(max_index)
    return result


# Domain size
domain_size = 10000

# Input size
input_size = 500

# number of hash function
m = 50

# Probability to set each bit of binary mask to 1.
p_1 = 0.5

# ---- Ideal world distribution ----
# Intersection size
z = 100

# probability that each hash function has a match, i.e., intersection size/union size.
# Not counting collision of non-matching values.
p_w = z / (2 * input_size - z)
print("p_w =", p_w)

r_values = list(range(m + 1))
dist_c = [binom.pmf(r, m, p_w * p_1) for r in r_values]
print_dist(dist_c)

# plt.bar(r_values, dist_c)
# plt.show()

# ---- Real world distribution ----
# Before seeing hash function. Assume uniformity on input set.
num_inputs_with_intersection_size(10000, 500)

# Pr[z = 0], ... Pr[z = input_size]


# Pr[w = 0], ... Pr[w = m]


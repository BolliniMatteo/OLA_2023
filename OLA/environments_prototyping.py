import numpy as np
import matplotlib.pyplot as plt
from environments import SingleClassEnvironment

"""
Just scripts to find some "good" functions for the environments
"""

def old_N(x):
    return np.sqrt((x ** 12) * 400)

def N(x):
    return np.sqrt((x ** 14) * 300)


def older_C(x):
    return 1/(0.01*(N(2.0)-N(x)+0.0001))*N(x)*0.7


def old_C(x):
    return 1/(0.007*(N(2.0)-N(x)))*N(x)*0.7


def C(x):
    return np.sqrt((x ** 16) * 200)


def create_single_class_environment(rng):
    # rng = np.random.default_rng(seed)
    return SingleClassEnvironment(N, lambda: rng.normal(0, 10, 1), C, ec, A, rng)



n_bids = 100
max_bid = 2.0

x = np.linspace(0.01, max_bid, n_bids)

plt.plot(x, N(x), label='number clicks')
plt.plot(x, C(x), label='payment')
plt.plot(x, N(x)*x, label='max payment')
plt.plot(x, 2.5*0.7*N(x)-C(x), label='reward')
plt.legend()
plt.show()

print(np.max(2.5*0.7*N(x)-C(x)))

# N ha un'immagine troppo vasta: se teniamo questo serve una noise con varianza dipendente dal valore di N
# OH, remember that N is an int, while C is a float!
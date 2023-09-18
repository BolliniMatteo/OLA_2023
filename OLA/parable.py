import numpy as np

"""
Simple code to compute the parameters of a parable given three points
"""

x1 = 100
x2 = 300
x3 = 700

y1 = 0.1
y2 = 0.6
y3 = 0.3

a = np.array([[x1**2, x1, 1],
              [x2**2, x2, 1],
              [x3**2, x3, 1]])
b = np.array([y1, y2, y3])

r = np.linalg.solve(a, b)
print("{}x^2 + {}x + {}".format(r[0], r[1], r[2]))
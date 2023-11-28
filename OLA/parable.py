import numpy as np

"""
Simple code to compute the parameters of a parable given three points
Simple plotting can be obtained with something like
https://www.wolframalpha.com/input?i=plot+-0.0005x%5E2+%2B+0.0050x+%2B+1.2000+range+30+to+50
"""

x1, y1 = 30, 0.9
x2, y2 = 40, 0.6
x3, y3 = 50, 0.2


a = np.array([[x1**2, x1, 1],
              [x2**2, x2, 1],
              [x3**2, x3, 1]])
b = np.array([y1, y2, y3])

r = np.linalg.solve(a, b)
print("%.4fx^2 + %.4fx + %.4f" % (r[0], r[1], r[2]))
print("Coefficients: (%.4f, %.4f, %.4f)" % (r[0], r[1], r[2]))

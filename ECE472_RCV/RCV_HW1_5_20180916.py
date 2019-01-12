# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 21:37:18 2018

@author: Shuyu
"""

import numpy as np
A = np.array([[1, 1.9, 3, 3.9, 5], [1,1,1,1,1]]).T
b = np.array([3.2,5,7.2,9.3,11.1]).T
q = np.linalg.lstsq(A, b)[0]
print("The parameter vector q determined by least squares estimation:")
print(q)

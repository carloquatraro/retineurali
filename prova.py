import numpy as np
a = np.array([1,2,3]).reshape(-1,1)
b = np.zeros([3,1])
c = np.ones([3,3])
d = np.column_stack((a,b,c))
print (d)
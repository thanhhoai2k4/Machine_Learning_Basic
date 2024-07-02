import numpy as np


# create array shape = (7,)
# print arr from index 1 to index (5-1) (n-1)
# we pass slice instead of index [start: end]
# we can also define the step: [start: end: step]
arr = np.array([1,2,3,4,5,6,7])
# print(arr[1:5])
# print(arr.shape)






import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[::2])
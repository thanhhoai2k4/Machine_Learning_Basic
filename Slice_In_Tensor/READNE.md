# slicing arrays 1-D

we pass slice instead of index like this: [start:end] \
we can also define the step: [start:end:step]

if we don't define start or end: default 0 with start or len(arr)-1 with end

## We can initialize array from 1 to 7 and show  array from index 1 to index 4
```

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5])
```

Output:
```
[2 3 4 5]
```


Show slice from index 4 to end

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[4:])
```
the output:
```
[5, 6, 7]
```



We can using negative slicing:

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[-3:-1])
```
define: index -2: is number 5(index 4), index -1: is index 6

the output:
```
    [5, 6]
```



Step:
```
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5:2])
```

the output:
```
[2 ,4]
```



note:
```
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[::2])
```

print all arr with step=2 ~ print(arr[0:7:2])




# Slicing 2_D arrays
see array arr: choose index 1 after slicing index from 1 to 4

```
import numpy as np

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[1,1:4])
```

note: arr[1:,1:4] : is define choose index 1:end after into it choose from index 1 to 4


From both elements, return index 2:
```
import numpy as np

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[0:2, 2])
```

the output: 
```
[3,8]
```
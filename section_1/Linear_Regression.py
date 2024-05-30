import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)

def f(x):
    return 3*x + 5

# create data
x = np.linspace(0,10,50).reshape(50,1) # numpy array
y = f(x)                  # numpy array

# show
#plt.plot(x, y, 'o')
#plt.show()
# add noice
def f_noice(x):
    noice = np.random.rand(50,1)
    return 3*(x+noice) + 5

y_noice = f_noice(x).reshape(50,1)


plt.subplot(2,1,1)
plt.plot(x,y, "o")
plt.plot(x,y, "r")
plt.xlabel("x")
plt.ylabel("y")
plt.title("y = 3x + 5")


plt.subplot(2,1,2)
plt.plot(x,y_noice, "o")
plt.xlabel("x")
plt.ylabel("y")
plt.title("y = 3(x+noice) + 5")
plt.savefig('figure.png')
plt.show()




bar = np.ones((50,1))
xbar = np.concatenate((bar, x), axis=1)

print(xbar.shape)
print(xbar)
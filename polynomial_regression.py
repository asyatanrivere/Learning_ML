import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20])
y=np.array([100,92,83,77,87,63,50,43,21,34,49,52,68,77,89,70,99,95])

my_model=np.poly1d(np.polyfit(x,y,3))
my_line=np.linspace(1,20,100)

plt.scatter(x,y)
plt.plot(my_line,my_model(my_line))
plt.show()
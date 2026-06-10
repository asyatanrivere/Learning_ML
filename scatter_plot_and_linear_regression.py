import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x=np.random.normal(6.0,2.0,1000)
y=np.random.normal(20.0,1.0,1000)

plt.scatter(x,y)
plt.show()

x_arr=np.array([2,4,6,7,8,10,13,11,24,26,47,52,62])
y_arr=np.array([99,98,96,82,88,77,72,56,61,60,51,88,40])

a,b,r,p,std_err=stats.linregress(x_arr,y_arr)
print(f"a:{a}")
print(f"b:{b}")
print(f"p:{p}")
"""
a:-0.6173028391167193
b:87.37741324921137
p:0.018270440333470032

p sonucun tesadüfi olup olmadığını ölçer
0.05ten büyükse ilişki tesadüf olabilir
"""

def myFunc(x):
    return a*x+b

guess=myFunc(10)
print(guess) # 81.20438485804418
# doğrusal regresyona göre tahmin yapar

my_model=list(map(myFunc,x_arr))
plt.scatter(x_arr,y_arr)
plt.plot(x_arr,my_model)
plt.show()
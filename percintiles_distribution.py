import numpy as np
import matplotlib.pyplot as plt

notes=np.array([10,20,30,40,50,60,70,80,90,100])

print(f"mean: {np.mean(notes)}")
print(f"Q1: {np.percentile(notes,25)}")
print(f"Q2: {np.percentile(notes,50)}")
print(f"Q3: {np.percentile(notes,75)}")

"""
mean: 55.0
Q1: 32.5
Q2: 55.0
Q3: 77.5"""

randoms=np.random.uniform(0,5,100)
print(randoms)
plt.hist(randoms,5)
plt.show()

normal_randoms=np.random.normal(7.0,2.0,100000) # average, standart deviation, size
plt.hist(normal_randoms,100)
plt.show()
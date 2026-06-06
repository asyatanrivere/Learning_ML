"""
standart deviation -> ortalama etrafındaki dağılımı gösterir, küçükse sayıların ortalama etrafında dağıldığını gösterir"""

import numpy as np

speed=np.array([50,55,60,65,70,75])
speed2=np.array([32,65,88,34,97,43])

print(f"\nMean speed: {np.mean(speed)}")
print(f"Population Std of speed: {np.std(speed,ddof=0)}")
print(f"Sample Std of speed: {np.std(speed,ddof=1)}")
print(f"Population Variance of speed: {np.var(speed,ddof=0)}")
print(f"Sample Variance of speed: {np.var(speed,ddof=1)}")
"""
Mean speed: 62.5
Population Std of speed: 8.539125638299666
Sample Std of speed: 9.354143466934854
Population Variance of speed: 72.91666666666667
Sample Variance of speed: 87.5
"""
print()

print(f"Mean speed2: {np.mean(speed2)}")
print(f"Population Std of speed2: {np.std(speed2,ddof=0)}")
print(f"Sample Std of speed2: {np.std(speed2,ddof=1)}")
print(f"Population Variance of speed2: {np.var(speed2,ddof=0)}")
print(f"Sample Variance of speed: {np.var(speed,ddof=1)}")
"""
Mean speed2: 59.833333333333336
Population Std of speed2: 25.58265471412657
Sample Std of speed2: 28.024394135585993
Population Variance of speed2: 654.4722222222223
Sample Variance of speed: 87.5
"""
"""
teorik hesaplamalarda -> varyans
yorum yapmak için -> standart sapma"""
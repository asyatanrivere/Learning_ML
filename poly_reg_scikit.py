import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
x=np.array([1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20])
y=np.array([100,99,95,87,77,62,63,50,43,40,44,59,62,77,87,95,99,101])

my_model=np.poly1d(np.polyfit(x,y,3))
# poly1d -> oluşan denklemi pyhton'da fonksiyon gibi kullanmamızı sağlar
# polfit -> şu an x ve y verilerinden 3 dereceli bir denklem oluşturur

print(r2_score(y,my_model(x)))
# polinom regresyonu için uygun olup olmadığını kontrol edeceğiz
# 0.8706967941972972 -> güçlü bir ilişki -> polinom regresyonu için uygun
# kullanmayı deneyelim

speed=my_model(4)
print(speed) # 84.8322667429599
speed2=my_model(7)
print(speed2) # 66.23788220129495

my_line=np.linspace(1,20,100)
plt.scatter(x,y)
plt.plot(my_line,my_model(my_line))
plt.show()
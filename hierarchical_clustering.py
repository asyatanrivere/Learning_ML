"""
hiyerarşik kümeleme benzer verileri birbirleriyle kümelemek için kullanılan bir yöntemdir Hiyerarşik kümeleme benzer verileri birbirlerine yakınlıklarına göre gruplayan bir tekniktir
küçük grupları birleştirerek bri ağaç yapısı kurar"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.cluster import AgglomerativeClustering

x=np.array([4,5,10,4,3,11,14,6,10,12])
y=np.array([21,19,24,17,16,25,24,22,21,21])

data=list(zip(x,y))
print(data)
"""
[(np.int64(4), np.int64(21)), (np.int64(5), np.int64(19)), (np.int64(10), np.int64(24)), (np.int64(4), np.int64(17)), (np.int64(3), np.int64(16)), (np.int64(11), np.int64(25)), (np.int64(14), np.int64(24)), (np.int64(6), np.int64(22)), (np.int64(10), np.int64(21)), (np.int64(12), np.int64(21))]
"""

linkage_data = linkage(data, method="ward", metric="euclidean") # öklidyen mesafe (metric)
# hiyerarşik kümeleme işlemini bu satırda yaptık
# ward -> ward yöntemi kümeleri birleştirirken toplam iç varyansı (hata kareleri toplamı) en az arttıracak şekilde gruplamamızı sağlıyor, birbirlerine en yakın olanları birleştirir.
# amaç, matematiksel olarak birbirine veya kümeye en yakın sayıları bulup gruplamak. Bu yüzden varyansa bakar.
dendrogram(linkage_data)
plt.show()


l_data=AgglomerativeClustering(n_clusters=2,linkage="ward")
# n_clusters -> kaç kümeye ayrılacağını gösterir
labels=l_data.fit_predict(data)
plt.scatter(x,y,c=labels)
plt.show()
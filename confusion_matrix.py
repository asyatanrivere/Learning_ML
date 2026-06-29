# modelin yaptığı doğru ve yanlış sınıflandırmaları gösteren bir tablodur

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

actual=np.random.binomial(1,0.9,size=1000)
predicted=np.random.binomial(1,0.9,size=1000)

c_matrix=metrics.confusion_matrix(actual,predicted)

cm=metrics.ConfusionMatrixDisplay(confusion_matrix=c_matrix,display_labels=[0,1])
cm.plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_plot.png")
plt.show()

"""
True Positive   TP    False Negative  FN
False Positive  FP    True Negative   TN
"""

accuracy=metrics.accuracy_score(actual,predicted)
# modelin ne sıklıkla doğru olduğunu söyler
# (TP+TN) / tüm tahminler
precision=metrics.precision_score(actual,predicted)
# pozitif tahminlerimizin içinde ne kadar doğru çıktığını gösterir, kaçırılmaması gereken durumlarda önemlidir (kanser tespiti)
# TP / (TP + FP)
sensitivity=metrics.recall_score(actual,predicted)
# gerçek pozitiflerin içinde ne kadarını bulabildiğini gösterir
# TP / (TP + FN)
specificity=metrics.recall_score(actual,predicted,pos_label=0) # recall
# özgürlük metriği: modelin gerçek negatiflerin içinde ne kadar doğru tahmin yaptığını gösterir
# po label = 0 bizim 0lara yönlendirecek
# TN / (TN + FP)
f1_score=metrics.f1_score(actual,predicted)
# precision ve recall dengesinin puanını ölçer. Puan yüksekse bu, modelin doğru tahminler yaptığını ve önemli şeyleri kaçırmadığını gösterir. Dengesiz veri setlerinde kullanılır (pozitif ve negatiflerin dengesiz dağıldığı veri setleri)
# 2 * (precision * recall)(precision + recall)

print(f"Accuracy: {accuracy}") # Accuracy: 0.812
# model 100 tahmin yaptıysa (evet ya da hayır) 80i doğru
print(f"Precision: {precision}") # Precision: 0.9039301310043668
# model 100 kişiye evet dediyse bunların 10u gerçekten doğru
print(f"Sensitivity (recall): {sensitivity}")
# Sensitivity (recall): 0.8964365256124721
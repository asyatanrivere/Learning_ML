import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt

df=pd.read_csv("data3.csv")

dN={"UK":0,"USA":1,"TR":2}
df["nationality"]=df["nationality"].map(dN)
dGo={"NO":0,"YES":1}
df["go"]=df["go"].map(dGo)

features=["age","experience","rank","nationality"]
x=df[features]
y=df["go"]

dtree=DecisionTreeClassifier()
dtree=dtree.fit(x,y)
# modeli eğitiyoruz

plt.figure(figsize=(15,9))
plot_tree(dtree,feature_names=features)
plt.savefig("decision_tree_plot.png")
plt.show()
"""
gini-> her zaman 0 ile 0.5 arasındadır. bölünmenin nasıl ayrıldığını gösterir
0--- örnekler saf hepsi aynı
0.5--- örnekler karışık
karar ağacı her adımda gini değerinin en az olduğu dallanmayı seçmek ister

values= kaç YES kaç NO olduğunu gösteriyor"""
new_data=pd.DataFrame([[40,10,7,1]],columns=["age","experience","rank","nationality"])
print(dtree.predict(new_data)) # [1]
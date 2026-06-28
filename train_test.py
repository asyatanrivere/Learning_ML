import pandas as pd
from sklearn.model_selection import train_test_split # train ve test yapmak için import ettik
from sklearn.linear_model import LinearRegression
# doğrusal regresyon yapmak için import ettik
from sklearn.metrics import r2_score

df=pd.read_csv("data.csv")
x=df[["Weight","Volume"]]
y=df["CO2"]

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

model=LinearRegression()
# modeli tanımladık
model.fit(x_train,y_train)
# modeli eğittik

y_predict=model.predict(x_test)
score=r2_score(y_test,y_predict)
# y_test ve y_predict karşılaştırılacak, uygun olup olmadığını görecez
# 1'e yakınsa çok uyumlu, 0'a yakınsa uyumlu değil
print(f"r2 score: {score}")
# r2 score: 0.32941109624012743
# not a good score

comparison=pd.DataFrame({"Gerçek CO2":y_test,"Tahmini CO2":y_predict})

print(comparison)
"""
    Gerçek CO2  Tahmini CO2
35         120   106.162301
13          94   101.152581
26         104   104.783275
30         115   106.137684
16          99   102.062149
31         117   106.843891
21          99   104.284220
12          99    97.453233
"""
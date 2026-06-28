import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler # for scale
from sklearn.preprocessing import StandardScaler # for z score

df=pd.read_csv("data.csv")
x=df[["Weight","Volume"]]
y=df["CO2"]

"""plt.scatter(df["Weight"],y)
plt.scatter(df["Volume"],y)
plt.show()"""

regr=linear_model.LinearRegression()
# doğrusal regresyon modelini oluşturduk
regr.fit(x,y)
# her ikisini de regresyona tabii tuttuk, makineyi eğitiyoruz
# iki bağımsız değişken gönderidğimiz için çoklu regresyon

new_data=pd.DataFrame([[2300,1300]],columns=["Weight","Volume"])
# tahmin etmesi için yeni değerler girdik

result=regr.predict(new_data)
print(f"new data: {new_data}")
# girdiğimiz değerlere göre CO2 salınımını tahmin etmesini istedik
print(result) # [107.2087328]
print(f"Intercept(b0): {regr.intercept_}")
print(f"Coefficients(b1,b2): {regr.coef_}")
"""
Intercept(b0): 79.69471929115939
Coefficients(b1,b2): [0.00755095 0.00780526]
                          W            v
"""
# -------------- SCALE ---------------------

notes=np.array([[40],[23],[90],[94],[78],[66],[80],[57]])

scaler=MinMaxScaler()

result=scaler.fit_transform(notes)
print(result)
"""
[[0.23943662]
 [0.        ]
 [0.94366197]
 [1.        ]
 [0.77464789]
 [0.6056338 ]
 [0.8028169 ]
 [0.47887324]]
 """

# --------------------- Z Score ---------------------

data={
    "Height":[160,158,178,156,183],
    "Weight":[56,42,79,48,83]
}

df_z=pd.DataFrame(data)
# önde data frame yapmamız lazım
scaler_z=StandardScaler()
# standartlaştırma normalizasyonu kullanacağız
result_z=scaler_z.fit_transform(df_z)
print(pd.DataFrame(result_z,columns=["Height","Weight"]))
"""
     Height    Weight
0 -0.624602 -0.339400
1 -0.803059 -1.187901
2  0.981517  1.054565
3 -0.981517 -0.824258
4  1.427661  1.296994
"""
scale=StandardScaler()
df2=pd.read_csv("data2.csv")
x2=df2[["Weight","Volume"]]
y2=df2["CO2"]

scaledx=scale.fit_transform(x2)
"""
[[-2.10389253 -1.59336644]
 [-0.55407235 -1.07190106]
 [-1.52166278 -1.59336644]
 [-1.78973979 -1.85409913]
 [-0.63784641 -0.28970299]
 .
 .
 """
regr2=linear_model.LinearRegression()
regr2.fit(scaledx,y2)

new_data2=pd.DataFrame([[2300,1.3]],columns=["Weight","Volume"])
# tahmin etmesi için yeni değerler girdik
scaled=scale.transform(new_data2)
result2=regr2.predict(scaled)
print(result2)
# [107.2087328]

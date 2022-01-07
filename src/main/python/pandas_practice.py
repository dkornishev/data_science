import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = {"Math": 60, "Science": 89, "English": 76, "Social Science": 86}

s1 = pd.Series(data)
print(s1)

s1["Русский"] = 99
print(s1)


X = np.arange(10,160,10)
Y = np.arange(1,12)
plt.hist(X,Y)
plt.show()
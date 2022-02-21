import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

cars = pd.DataFrame(data = [["bmw", 8, 3000], ["mcds", 6, 3500], ["жигули", 4, 2500], ["bmw2", 8, 3200], ["mcds2", 6, 4500], ["жигули2", 4, 3700]], columns=["car", "cyl", "wt"])

print(cars)

sns.stripplot(cars['wt'],cars['cyl'])
plt.show()

sns.boxplot(cars["cyl"], cars["wt"])
plt.show()

b = 3.1
print(type(b))
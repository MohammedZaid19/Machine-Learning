import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

xpoints = np.array([1,8])
ypoints = np.array([3,10])
plt.plot(xpoints,ypoints)

y = np.array([35,20,50,40])
plt.pie(y)

sns.displot([0,1,23,4,5,6,7])
plt.show()
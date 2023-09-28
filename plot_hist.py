from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
df = pd.read_csv(r'c:\Users\vikto\Downloads\scatterXY_A.csv')
filepath = r'C:\Users\vikto\Downloads\scatterXY_C.csv'





# Creating dataset
x = df['x']
y = df['y']
# Creating histogram
#plt.bar(b,a)
#plt.show()

# Create a histogram using Seaborn
#sns.histplot(b, kde=True, color='skyblue')

# Add labels and title
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.title('Histogram with KDE (Kernel Density Estimation)')

# Show the plot
plt.plot(x,np.random.random(0,1),'o')

plt.show()
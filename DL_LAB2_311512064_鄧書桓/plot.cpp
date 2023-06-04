import matplotlib.pyplot as plt
import numpy as np

# read in the csv file
data = np.genfromtxt('mean_score.csv', delimiter=',')

index = np.arange(data.shape[0])

# plot the data
plt.plot((index+1)* 1000, data, label='lr : 0.1')


plt.xlabel('Epoch')
plt.ylabel('Mean Score')
plt.title('Mean Score vs Epoch')
plt.legend()

plt.savefig('mean_score.png')


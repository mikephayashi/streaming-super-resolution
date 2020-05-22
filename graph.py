import matplotlib.pyplot as plt

import csv
x = []
y = []
iteration = 50
with open('./results/vqvae_params.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',',)
    for row in spamreader:
        for val in row:
            y.append(float(val))
            x.append(iteration)
            iteration += 50

plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
plt.scatter(x, y)
plt.show()

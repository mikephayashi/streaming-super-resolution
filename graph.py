import matplotlib.pyplot as plt

import csv
x = []
y = []
iteration = 10
with open('/Users/michaelhayashi/Desktop/GCP/VAE/losses.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',',)
    for row in spamreader:
        for val in row:
            # if float(val) < 1:
            y.append(float(val))
            x.append(iteration)
            iteration += 10


tick = 0.1
ticks = []
for i in range(0, 11):
    ticks.append(tick)
    tick += 0.1

plt.yticks(ticks)
plt.scatter(x, y)
plt.show()

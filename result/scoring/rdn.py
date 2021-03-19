import random
import csv

output = []
for i in range(17):
    output.append([random.randint(0, 1) for i in range(10)])

with open('random.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(output)
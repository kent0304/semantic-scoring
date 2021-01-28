import csv 

with open('PictureDescribingA.csv') as f:
    reader = csv.reader(f)
    result_A = [row[2:] for row in reader]
    result_A[0][0] = 'ability'
    result_A[0][1] = 'mother_tongue'
    for i in range(10):
        result_A[0][i+2] = 'question{}'.format(i+1)

with open('answers_A.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(result_A)

with open('PictureDescribingB.csv') as f:
    reader = csv.reader(f)
    result_B = [row[2:] for row in reader]
    result_B[0][0] = 'ability'
    result_B[0][1] = 'mother_tongue'
    for i in range(10):
        result_B[0][i+2] = 'question{}'.format(i+1)

with open('answers_B.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(result_B)

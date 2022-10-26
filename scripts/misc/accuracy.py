import sys

def getData(path):
    data = []
    for line in open(path):
        if line[0] != '#' and len(line) > 2:
            data.append(line.split('\t')[4])
    return data

data1 = getData(sys.argv[1])
data2 = getData(sys.argv[2])
print(len(data1), len(data2))
print(sum([x1 == x2 for x1, x2 in zip(data1, data2)])/len(data1))


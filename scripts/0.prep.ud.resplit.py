import myutils
import os

udDir = 'data/ud-treebanks-v2.2/'

def getSize(path):
    if os.path.isfile(path):
        return sum([int(len(line) < 3) for line in open(path)])
    return 0

def readConll(path):
    data = [[]]
    for line in open(path):
        if len(line) < 3:
            data.append([])
        else:
            data[-1].append(line)
    return data[:-1] # remove last empty one

def resize(origPath, lastN, newPath):
    data = readConll(origPath)
    newFile = open(newPath, 'w')
    for conlSent in data[-lastN:]:
        newFile.write(''.join(conlSent) + '\n')
    newFile.close()
    newOrigFile = open(origPath, 'w')
    for conlSent in data[:-lastN]:
        newOrigFile.write(''.join(conlSent) + '\n')
    newOrigFile.close()
    

for treebankDir in os.listdir(udDir):
    trainPath, devPath, testPath = myutils.getTrainDevTest(udDir + treebankDir)
    devSize = getSize(devPath)

    # first get more dev data
    if devPath == '' and trainPath != '':
        trainSize = getSize(trainPath)
        if trainSize > 100:        
            devSize = min(100, trainSize-100)
            # take last 100 from train for dev
            devPath = testPath.replace('test','dev')
            resize(trainPath, devSize, devPath)
            
    # now get tune data from dev
    if devSize != 0:
        tuneSize = int(devSize/3)
        # take last tuneSize from dev for tune
        resize(devPath, tuneSize, testPath.replace('test', 'tune'))


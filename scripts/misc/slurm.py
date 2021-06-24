import sys
import os

if len(sys.argv) < 4:
    print('please provide: path, job name, and time (hours)')
    exit(1)

run = True

def makeFile(name, task, idx, time):
    name = name + '.' + str(idx)
    outFile = open(name + '.job', 'w')
    outFile.write('#!/bin/bash\n')
    outFile.write('\n')
    outFile.write('#SBATCH --job-name=' + name + '\n')
    outFile.write('#SBATCH --output=' + name + '.out\n')
    outFile.write('#SBATCH --cpus-per-task=2\n')
    outFile.write('#SBATCH --time=' + time + ':00:00\n')
    outFile.write('#SBATCH --gres=gpu\n')
    outFile.write('#SBATCH --mem=88G\n')
    outFile.write('#SBATCH --mail-type=BEGIN,END,FAIL\n')
    #outFile.write('#SBATCH partition=brown')
    outFile.write('#SBATCH --partition=red\n')

    outFile.write('\n')
    outFile.write(task)
    if run:
        cmd = 'sbatch ' + name + '.job'
        print(cmd)
    outFile.close()
    cmd = 'sed -i "s;device [0-9];device \$CUDA_VISIBLE_DEVICES;g" ' + name + '.job'
    os.system(cmd)

jobSize = 1
if len(sys.argv) > 4:
    jobSize = int(sys.argv[4])
concat = ''
counter = 0
for lineIdx, line in enumerate(open(sys.argv[1])):
    if len(line) < 2:
        continue
    if lineIdx % jobSize == 0 and concat != '':
        makeFile(sys.argv[2], concat, counter, sys.argv[3])
        counter += 1
        concat = ''
    concat += line
if concat != '':
    makeFile(sys.argv[2], concat, counter, sys.argv[3])


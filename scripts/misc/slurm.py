import os
import sys

if len(sys.argv) < 4:
    print('please provide: path, job name, and time (hours)')
    exit(1)

run = True


def make_file(name, task, idx, time):
    name = name + '.' + str(idx)
    out_file = open(name + '.job', 'w')
    out_file.write('#!/bin/bash\n')
    out_file.write('\n')
    out_file.write('#SBATCH --job-name=' + name + '\n')
    out_file.write('#SBATCH --output=' + name + '.out\n')
    out_file.write('#SBATCH --cpus-per-task=2\n')
    out_file.write('#SBATCH --time=' + time + ':00:00\n')
    out_file.write('#SBATCH --nodes=1\n')
    out_file.write('#SBATCH --nodelist=cn[3,4,5,6,7,9,10,12,13,18,19]\n')
    out_file.write('#SBATCH --gres=gpu\n')
    out_file.write('#SBATCH --mem=20G\n')
    out_file.write('#SBATCH --mail-type=BEGIN,END,FAIL\n')
    out_file.write('#SBATCH --partition=scavenge\n')

    out_file.write('\n')
    out_file.write('module load Python/3.12.3-GCCcore-13.3.0\n\n')
    out_file.write(task)
    if run:
        cmd = 'sbatch ' + name + '.job'
        print(cmd)
    out_file.close()
    #cmd = 'sed -i "s;device [0-9];device \$CUDA_VISIBLE_DEVICES;g" ' + name + '.job'
    #os.system(cmd)


jobSize = 1
if len(sys.argv) > 4:
    jobSize = int(sys.argv[4])
concat = ''
counter = 0
for lineIdx, line in enumerate(open(sys.argv[1])):
    if len(line) < 2:
        continue
    if lineIdx % jobSize == 0 and concat != '':
        make_file(sys.argv[2], concat, counter, sys.argv[3])
        counter += 1
        concat = ''
    concat += line
if concat != '':
    make_file(sys.argv[2], concat, counter, sys.argv[3])

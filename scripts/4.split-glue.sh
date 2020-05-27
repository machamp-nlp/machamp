#!/bin/bash
#     8551     74363    373706 CoLA.train
#   392702  12088478  71870741 MNLI.train
#     3668    142632    863141 MRPC.train
#   108436   4127981  26430673 QNLI.train
#   363849   8414393  45079484 QQP.train
#     2490    132856    834854 RTE.train
#   549367  11682357  64138792 SNLI.train
#    67349    701073   3806066 SST-2.train
#      635     18291     96949 WNLI.train
mkdir -p ../data/glue/learnc
for file in MNLI QNLI QQP SNLI SST-2
do 
    for k in 5000 10000 20000 50000 100000
    do
	head -$k ../data/glue/$file.train > ../data/glue/learnc/$file-$k.train
    done
done

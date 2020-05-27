#MNLI QNLI QQP SNLI SST-2
cd ../configs
mkdir -p learnc
for name in mnli qnli qqp snli 
do
    for k in 5000 10000 20000 50000 100000
    do     
	upper=`echo $name | tr [a-z] [A-Z]`
	cat glue.$name.json |  sed "s/$upper.train/learnc\/$upper-${k}.train/" > learnc/glue.$name-$k.json
    done
done
# sst
for k in 5000 10000 20000 50000 100000
do
    cat glue.sst.json |  sed "s/SST-2.train/learnc\/SST-2-${k}.train/" > learnc/glue.sst-$k.json
done
cd -

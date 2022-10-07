mkdir -p data
mkdir data/nlu
curl -O -J -L https://fb.me/multilingual_task_oriented_data
mv multilingual_task_oriented_data data/nlu/
cd data/nlu
unzip multilingual_task_oriented_data
cd ../../


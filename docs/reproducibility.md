### Reproducibility
[back to main README](../README.md)

To reproduce all the results from our EACL 2021 paper, all one has to do is run:

```
git clone https://github.com/machamp-nlp/machamp.git
cd machamp
pip3 install requirements.txt
# now collect all hidden datasets (this is the hardest part),
# or be ok with leaving them out of the experiments
./scripts/runAll.sh
```

In practice however, this would take very very long. So we suggest to pick the experiment 
you are interested in from the `runAll.sh` file, and manually run it multi-threaded with e.g. `parallel` or in `SLURM`. The experiments are numbered, so you can see which scripts to run for each part in `runAll.sh.

We also provide all the predictions of our model in a `.tar.gz` archive, it can be downloaded at: http://itu.dk/people/robv/data/machamp/preds.tar.gz 

All scripts starting with `5.` can be used to generate the tables/graphs in the paper.


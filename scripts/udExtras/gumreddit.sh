cd gum
python3 _build/process_reddit.py
python3 _build/build_gum.py

cd _build/target/dep/not-to-release
cat GUM_reddit_callout.conllu GUM_reddit_card.conllu GUM_reddit_conspiracy.conllu GUM_reddit_gender.conllu GUM_reddit_introverts.conllu GUM_reddit_polygraph.conllu GUM_reddit_racial.conllu GUM_reddit_ring.conllu GUM_reddit_social.conllu GUM_reddit_space.conllu GUM_reddit_stroke.conllu GUM_reddit_superman.conllu > /data/rob/gum.train.conllu
cat GUM_reddit_macroeconomics.conllu GUM_reddit_pandas.conllu GUM_reddit_steak.conllu > /data/rob/gum.dev.conllu
cat GUM_reddit_bobby.conllu GUM_reddit_escape.conllu GUM_reddit_monsters.conllu > /data/rob/gum.test.conllu


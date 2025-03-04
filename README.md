# LLM-Ordering-Preferences

Github repo for the assessment of ordering preferences in LLMs.

This repo is a mess and we are currently in the process of cleaning it up, but it may take some time. If you have specific questions feel free to reach out to us.

Dolma_corpus_creation.py is used to create a onegram, twogram, and threegram cropus. Warning that this can take a long time to run, upwards of several weeks depending on your computational power.

Dolma_corpus_revise.py checks to see if every file has been downloaded, and downloads them if they haven't.

corpus_search.py takes a .csv file and gathers the frequency for that n-gram. 

reading-times-and-ordering-prefs.Rmd was used to get ordering prefs of binomials for different LLMs. 

nonce-binoms_analysis.Rmd was used to analyze the ordering preferences for novel binomials.

Results are forthcoming.

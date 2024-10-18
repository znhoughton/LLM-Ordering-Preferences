from collections import defaultdict
from collections import Counter
import json
import re
import multiprocessing as mp
from os import listdir
from os.path import isfile, join
import gzip
import time
import glob
import pandas as pd
import csv
import os
import re
import concurrent.futures
import datetime
from multiprocessing import set_start_method
from multiprocessing import get_context
import logging
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, TimeoutError, as_completed, ALL_COMPLETED
from pebble import ProcessPool, ProcessExpired
import sys

###need to test if timeout works if set to e.g., 5mins


def onegram(sentence):
	text = re.sub(r'[^\w\s]', '', sentence).lower() #lowercase and strip punctuation
	words = text.split()
	return [word for word in words if len(word) < 40]

def bigrams(sentence):
    text = re.sub(r'[^\w\s]', '', sentence).lower()
    words = text.split()
    words = [word for word in words if len(word) < 40]
    return zip(words, words[1:])

def trigrams(sentence):
     text = re.sub(r'[^\w\s]', '', sentence).lower()
     words = text.split()
     words = [word for word in words if len(word) < 40]
     return zip(words, words[1:], words[2:])


# this is for parallel processing


def process_individual_file(gzip_file):
    now = datetime.datetime.now()
    print(f"Currently Processing: {gzip_file} at: {now}", flush=True)
    
    three_gram_ind_counter = Counter()
    
    with gzip.open(gzip_file, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                three_gram_ind_counter.update(trigrams(line))
            except EOFError:
                print(gzip_file, ' is corrupted')
    
    write_file_to_csv(three_gram_ind_counter, gzip_file, 'trigram_files')
    now = datetime.datetime.now()
    print(f"Finished writing {gzip_file} at: {now}", flush = True)


def process_gzip_file_parallel(gzip_files, num_workers=5, timeout=30000):
    with ProcessPool(max_workers=num_workers) as pool:
        futures = {pool.schedule(process_individual_file, args=(file,), timeout=timeout): file for file in gzip_files}
        while futures:
            # Wait for at least one future to complete
            completed, not_done = wait(futures, timeout=timeout, return_when=FIRST_COMPLETED)
            for future in completed:
                file = futures.pop(future)  # Remove the completed future from the futures set
                try:
                    result = future.result()
                    # results.append(result)  # You can collect results here if needed
                    print(f"File {file} processed successfully.")
                except (ProcessExpired, RuntimeError, TimeoutError) as error:
                    print(f"Error {error} processing file {file}: {str(error)}")
            
            # Continue with the futures that are not done yet
            futures = {future: futures[future] for future in not_done}


def write_file_to_csv(counter_file, file, ngram_type):
       file_name = (os.path.splitext(file)[0]).split('/')[-1]
       now = datetime.datetime.now()
       print(f"Currently Writing: {file_name} at {now}, saved in ./{ngram_type}/{file_name}", flush=True)
       os.makedirs(f'./{ngram_type}', exist_ok=True)
       with gzip.open(f'./{ngram_type}/{file_name}.csv.gz', 'wt') as csvfile:
            fieldnames = ['ngram', 'count']
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)
            for k,v in counter_file.items():
                if isinstance(k, str):
                    k = [k]
                ngram_identity = '\t'.join(k)
                writer.writerow([ngram_identity, v])
           
def process_onegram_files():
    print("Currently processing onegram_files", flush=True)
    onegram_files = glob.glob('./onegram_files/*.csv.gz')
    intermediate_dfs = []
    batch_size = 100
    
    for i in range(0, len(onegram_files), batch_size):
        now = datetime.datetime.now()
        print(f"Currently processing batch {i // batch_size + 1} / {len(onegram_files) // batch_size + 1} at: {now}")
        batch = onegram_files[i:i + batch_size]
        batch_dfs = [pd.read_csv(file, compression='gzip', encoding = 'utf-8') for file in batch]
        batch_df = pd.concat(batch_dfs).groupby('ngram', as_index=False).sum()
        intermediate_dfs.append(batch_df)
    
    result_df = pd.concat(intermediate_dfs).groupby('ngram', as_index=False).sum()
    result_df = result_df.sort_values(by=['count'], ascending=False)
    result_df.to_csv('full_onegram_corpus.csv.gz', index=False, compression = 'gzip')
    
def process_bigram_files():
    print("Currently processing bigram_files", flush=True)
    bigram_files = glob.glob('./bigram_files/*.csv.gz')
    intermediate_dfs = []
    batch_size = 100
    
    for i in range(0, len(bigram_files), batch_size):
        now = datetime.datetime.now()
        print(f"Currently processing batch {i // batch_size + 1} / {len(bigram_files) // batch_size + 1} at: {now}")
        batch = bigram_files[i:i + batch_size]
        batch_dfs = [pd.read_csv(file, compression='gzip', encoding = 'utf-8') for file in batch]
        batch_df = pd.concat(batch_dfs).groupby('ngram', as_index=False).sum()
        intermediate_dfs.append(batch_df)
    
    result_df = pd.concat(intermediate_dfs).groupby('ngram', as_index=False).sum()
    result_df = result_df.sort_values(by=['count'], ascending=False)
    result_df.to_csv('full_bigram_corpus.csv.gz', index=False, compression = 'gzip')
    
    
def process_trigram_files():
    print("Currently processing trigram_files", flush=True)
    trigram_files = glob.glob('./trigram_files/*.csv.gz')
    intermediate_dfs = []
    batch_size = 2
    
    for i in range(0, len(trigram_files), batch_size):
        now = datetime.datetime.now()
        print(f"Currently processing batch {i // batch_size + 1} / {len(trigram_files) // batch_size + 1} at: {now}")
        batch = trigram_files[i:i + batch_size]
        batch_dfs = [pd.read_csv(file, compression='gzip', encoding = 'utf-8') for file in batch]
        batch_df = pd.concat(batch_dfs).groupby('ngram', as_index=False).sum()
        intermediate_dfs.append(batch_df)
    
    result_df = pd.concat(intermediate_dfs).groupby('ngram', as_index=False).sum()
    result_df = result_df.sort_values(by=['count'], ascending=False)
    result_df.to_csv('full_trigram_corpus.csv.gz', index=False, compression = 'gzip')


def check_and_process_trigrams_corpus(): #for some reason sometimes all of the files aren't being processed, so this script will check to see if they've been downloaded and try again if they haven't. I wonder if there's a better way, since the process_gzip_file_parallel() function seems to slow down significantly after a few hours. I wonder if manually restarting it after a certain number of hours would increase the speed of processing the files. 
    path_source = 'Dolma/'
    path_destination = 'trigram_files/'
    #gzip_files_source = [(path_source + f) for f in listdir(path_source) if isfile(join(path_source, f))]	#all the gzip files in the directory
    gzip_files_source = [f.split('.')[0] for f in listdir(path_source)]
    #gzip_files_destination = [(path_destination + f) for f in listdir(path_destination) if isfile(join(path_destination, f))]
    gzip_files_destination = [f.split('.')[0] for f in listdir(path_destination)]
    if len(gzip_files_source) == len(gzip_files_destination):
        print(f'Number of files in source match the number of files in destination: {len(gzip_files_source)}\nProcessing files now')
        process_trigram_files()
        return True #to break the while loop if everything downloaded correctly
    else:
        files_not_downloaded = list(set(gzip_files_source) - set(gzip_files_destination))
        print(f'{len(files_not_downloaded)} files not downloaded. Attempting to download them.')
        
        files_to_process = [path_source + filename + '.json.gz' for filename in files_not_downloaded]
        process_gzip_file_parallel(files_to_process)
        
        
def main():
    if __name__ == "__main__":
            #set_start_method("spawn")
            t1 = time.perf_counter()
            path = 'Dolma/'
            check_and_process_trigrams_corpus()    
            t2 = time.perf_counter()
            print(t2 - t1)
            #check to make sure the number of gzip_files created are equal to the number of files in the path
            
        
        

#def main_serial():
#    if __name__ == "__main__":
#        t1 = time.perf_counter()
#        path = 'Dolma/'
#        gzip_files = [(path + f) for f in listdir(path) if isfile(join(path, f))]	#all the gzip files in the directory
#        for i in gzip_files:
#            print('Current file: ', i)
#            process_gzip_file_serial(i)
#        write_result_to_file_serial()
#        t2 = time.perf_counter()
#        print(t2 - t1)



t1 = time.perf_counter()
main()
t2 = time.perf_counter()
t2 - t1




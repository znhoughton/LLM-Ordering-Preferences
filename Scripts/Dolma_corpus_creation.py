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
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

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


def process_individual_file_onegram(gzip_file):
    now = datetime.datetime.now()
    print(f"Currently Processing: {gzip_file} at: {now}", flush=True)
    
    one_gram_ind_counter = Counter()
    
    with gzip.open(gzip_file, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                one_gram_ind_counter.update(onegram(line))
            except EOFError:
                print(gzip_file, ' is corrupted')
    
    write_file_to_csv(one_gram_ind_counter, gzip_file, 'onegram_files')
    now = datetime.datetime.now()
    print(f"Finished writing {gzip_file} at: {now}", flush = True)

def process_individual_file_bigram(gzip_file):
    now = datetime.datetime.now()
    print(f"Currently Processing: {gzip_file} at: {now}", flush=True)
    
    two_gram_ind_counter = Counter()
    
    with gzip.open(gzip_file, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                two_gram_ind_counter.update(bigrams(line))
            except EOFError:
                print(gzip_file, ' is corrupted')
    
    write_file_to_csv(two_gram_ind_counter, gzip_file, 'bigram_files')
    now = datetime.datetime.now()
    print(f"Finished writing {gzip_file} at: {now}", flush = True)
    
def process_individual_file_trigram(gzip_file):
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
    

def process_gzip_file_parallel(ngram_type, gzip_files, num_workers=50, timeout=30000):
    with ProcessPool(max_workers=num_workers) as pool:
        
        processing_functions = {
        'onegram': process_individual_file_onegram,
        'twogram': process_individual_file_bigram,
        'threegram': process_individual_file_trigram
        }
        if ngram_type in processing_functions:
            process_file_function = processing_functions[ngram_type]  # Call the corresponding function
        else:
            raise ValueError("Invalid n-gram type specified.")
        
        futures = {pool.schedule(process_file_function, args=(file,), timeout=timeout): file for file in gzip_files}
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


# def process_file(file):
    # print(f"Processing: {file}")
    #Read the gzip file in chunks to avoid high memory usage
    # df = pd.read_csv(file, compression='gzip', encoding='utf-8', chunksize=100000000)  # Adjust chunksize as necessary
    # return df


           
# def process_onegram_files():
    # onegram_files = glob.glob('./onegram_files/*.csv.gz')
    # final_counts = defaultdict(int)  # Use defaultdict to simplify summation of counts

    # for file in onegram_files:
        #print(f"Processing {file}...")
        # for chunk in process_file(file):
            # for index, row in chunk.iterrows():
                # final_counts[row['ngram']] += row['count']  # Sum counts for each unique n-gram

    #Convert the defaultdict back to a DataFrame
    # final_result = pd.DataFrame(final_counts.items(), columns=['ngram', 'count'])

    #Save the final result to a single CSV file
    # final_result.to_csv('full_onegram_corpus.csv.gz', index=False, compression='gzip') 
    
# def process_bigram_files():
    # bigram_files = glob.glob('./bigram_files/*.csv.gz')
    # final_counts = defaultdict(int)  # Use defaultdict to simplify summation of counts

    # for file in bigram_files:
        #print(f"Processing {file}...")
        # for chunk in process_file(file):
            # for index, row in chunk.iterrows():
                # final_counts[row['ngram']] += row['count']  # Sum counts for each unique n-gram

    #Convert the defaultdict back to a DataFrame
    # final_result = pd.DataFrame(final_counts.items(), columns=['ngram', 'count'])

    #Save the final result to a single CSV file
    # final_result.to_csv('full_bigram_corpus.csv.gz', index=False, compression='gzip') 
 
# def process_trigram_files():
    # trigram_files = glob.glob('./test_trigram_files/*.csv.gz')
    # final_counts = defaultdict(int)  # Use defaultdict to simplify summation of counts

    # for file in trigram_files:
        #print(f"Processing {file}...")
        # for chunk in process_file(file):
            # for index, row in chunk.iterrows():
                # final_counts[row['ngram']] += row['count']  # Sum counts for each unique n-gram

    #Convert the defaultdict back to a DataFrame
    # final_result = pd.DataFrame(final_counts.items(), columns=['ngram', 'count'])

    #Save the final result to a single CSV file
    # final_result.to_csv('full_trigram_corpus.csv.gz', index=False, compression='gzip') 
    

# def process_ngram_files(ngram_type):
    # processing_functions = {
        # 'onegram': process_onegram_files,
        # 'twogram': process_bigram_files,
        # 'threegram': process_trigram_files
        
    # if ngram_type in processing_functions:
        # processing_functions[ngram_type]()  # Call the corresponding function
    # else:
        # raise ValueError("Invalid n-gram type specified.")
        




def check_and_process_corpus(ngram = 'onegram'): #for some reason sometimes all of the files aren't being processed, so this script will check to see if they've been downloaded and try again if they haven't. I wonder if there's a better way, since the process_gzip_file_parallel() function seems to slow down significantly after a few hours. I wonder if manually restarting it after a certain number of hours would increase the speed of processing the files. 
    path_source = 'Dolma/'
    path_destination = f'{ngram}_files/'
    os.makedirs(path_destination, exist_ok=True)
    #gzip_files_source = [(path_source + f) for f in listdir(path_source) if isfile(join(path_source, f))]	#all the gzip files in the directory
    gzip_files_source = [f.split('.')[0] for f in listdir(path_source)]
    #gzip_files_destination = [(path_destination + f) for f in listdir(path_destination) if isfile(join(path_destination, f))]
    gzip_files_destination = [f.split('.')[0] for f in listdir(path_destination)]
    files_not_downloaded = list(set(gzip_files_source) - set(gzip_files_destination))
    if len(files_not_downloaded) == 0:
            print(f'Files downloaded properly')
            #process_ngram_files(ngram_type = ngram)
            #return True #to break the while loop if everything downloaded correctly
    else:
        print(f'{len(files_not_downloaded)} files not downloaded. Attempting to download them.')
        files_to_process = [path_source + filename + '.json.gz' for filename in files_not_downloaded]
        process_gzip_file_parallel(ngram_type='onegram', gzip_files=files_to_process)
        
        if len(files_not_downloaded) == 0:
            print(f'Files downloaded properly\nProcessing Now')
            #process_ngram_files(ngram_type = ngram) #we're going to keep them as separate csvs, for memory reasons 
        
def main():
    if __name__ == "__main__":
            #set_start_method("spawn")
            t1 = time.perf_counter()
            path = 'Dolma/'
            check_and_process_corpus(ngram = 'onegram')  #onegram, twogram, or threegram -- admittedly a bit confusing, should change to onegram, bigram, trigram later
            t2 = time.perf_counter()
            print(t2 - t1)
            #check to make sure the number of gzip_files created are equal to the number of files in the path
            
        

t1 = time.perf_counter()
main()
t2 = time.perf_counter()
t2 - t1




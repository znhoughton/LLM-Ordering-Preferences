#bigram counts --> (count(bread) / corpus_size) * (count(bread and) / count(bread)) * (count(and butter) / count(and))
#				   (count(butter / corpus_size) * (count(butter and) / count(butter)) * (count(and bread) / count(and))
#corpus_size = 1560674367427
#count(and) = 

import pandas as pd
import glob
import concurrent.futures
from functools import partial
import time
import logging
from tqdm import tqdm  # Import tqdm for the progress bar
from collections import defaultdict
import csv




logging.basicConfig(
    level=logging.DEBUG,  # Capture all levels of logs (DEBUG and above)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Include timestamp, level, and message
    handlers=[
        logging.FileHandler("get_bigram_counts.log"),  # Save logs to file
        logging.StreamHandler()  # Also print logs to console
    ]
)

def process_file_for_trigrams(file, trigrams, chunk_size=40000000):
    # Initialize a dictionary to hold word frequencies for each trigram
    word_freqs = defaultdict(int)

    # Check if the file exists and is not empty
    if not os.path.exists(file):
        logging.error(f"File not found: {file}")
        return {}, file  # Return empty dictionary and filename

    if os.path.getsize(file) == 0:
        logging.warning(f"Skipping empty file: {file}")
        return {}, file  # Return empty dictionary and filename

    try:
        # Read the file in chunks
        for chunk in pd.read_csv(file, compression='gzip', encoding='utf-8', chunksize=chunk_size, usecols=['ngram', 'count']):
            if chunk.empty:
                logging.warning(f"File {file} has no valid data.")
                continue  # Skip to the next chunk

            # Filter rows where 'ngram' is in the list of trigrams
            filtered_chunk = chunk[chunk['ngram'].isin(trigrams)]

            # Sum the counts for each trigram in the chunk
            for trigram, group in filtered_chunk.groupby('ngram'):
                word_freqs[trigram] += group['count'].sum()

        return word_freqs, file  # Return the dictionary of frequencies and the file name

    except pd.errors.EmptyDataError:
        logging.error(f"EmptyDataError: No columns to parse from file {file}. Skipping.")
    except FileNotFoundError:
        logging.error(f"FileNotFoundError: File {file} was not found.")
    except OSError as e:
        logging.error(f"OSError: Issue reading file {file} - {e}")
    except Exception as e:
        logging.error(f"Unexpected error processing file {file}: {e}")

    return {}, file  # Return an empty dictionary in case of errors
	
	
def trigrams_search_parallel(ngrams, ngram_type = 'threegram', chunk_size=10000000, num_workers=None): #ngram_type can be onegram, twogram, or threegram

    ngram_dirs = {
        'onegram': 'onegram_files',
        'twogram': 'bigram_files',
        'threegram': 'trigram_files'
        }

    file_dir = ngram_dirs[ngram_type]
    files = glob.glob(f'./{file_dir}/*.csv.gz')

    # Initialize a dictionary to hold total frequencies for all trigrams
    total_word_freqs = defaultdict(int, {trigram: 0 for trigram in ngrams})

    # Create a partial function with pre-defined arguments
    process_with_args = partial(process_file_for_trigrams, trigrams=ngrams, chunk_size=chunk_size)

    # Use ProcessPoolExecutor to parallelize the processing of files
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # List to hold futures
        futures = {executor.submit(process_with_args, file): file for file in files}

        # Use tqdm to create a progress bar for the file processing
        for future in concurrent.futures.as_completed(futures):
            with tqdm(total=len(futures), desc="Searching files") as progress_bar:
                    word_freqs, file_name = future.result()  # Get the result and file name

                    # Accumulate frequencies for each trigramtri
                    for ngram, freq in word_freqs.items():
                        total_word_freqs[ngram] += freq

                    logging.info(f"Processed file: {file_name}")  # Log the processed file
                    progress_bar.update(1)  # Update the progress bar

    return total_word_freqs  # Return the dictionary of total frequencies



def get_wordcount_file(file, chunk_size=40000000):
    corpus_size = defaultdict(int)
    for chunk in pd.read_csv(file, compression='gzip', encoding='utf-8', usecols=['count'], chunksize = 10000000):
        corpus_size[file] += chunk['count'].sum()
    print(f"File {file} size is: {corpus_size}")
    return corpus_size, file
        
        


def get_corpus_size(ngram_type='onegram'):
    ngram_dirs = {
        'onegram': 'onegram_files',
        'twogram': 'bigram_files',
        'threegram': 'trigram_files'
    }
    corpus_size = 0
    file_dir = ngram_dirs[ngram_type]
    files = glob.glob(f'./{file_dir}/*.csv.gz')
    num_workers = 100
    # Initialize a dictionary to hold total frequencies for all trigrams
    total_word_freqs = defaultdict(int, {file: 0 for file in files})

    # Create a partial function with pre-defined arguments
    process_with_args = partial(get_wordcount_file, chunk_size=10000000)

    # Use ProcessPoolExecutor to parallelize the processing of files
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # List to hold futures
        futures = {executor.submit(process_with_args, file): file for file in files}

        # Use tqdm to create a progress bar for the file processing
        for future in concurrent.futures.as_completed(futures):
            with tqdm(total=len(futures), desc="Searching files") as progress_bar:
                    word_freqs, file_name = future.result()  # Get the result and file name

                    # Accumulate frequencies for each trigram
                    for filename, freq in word_freqs.items():
                        corpus_size += freq

                    logging.info(f"Processed file: {file_name}")  # Log the processed file
                    progress_bar.update(1)  # Update the progress bar

    return corpus_size  # Return the dictionary of total frequencies



def get_bigram_counts(): #get counts of Word1 and, and Word2, Word2 and, and Word1
	binomials_data = pd.read_csv('binomials.csv')


	num_workers=60
	alpha_bigrams = binomials_data['Word1'] + ' and' #need to get all the 'bread and' and 'and butter'
	alpha_bigrams2 = 'and ' + binomials_data['Word2']
	
	alpha_bigrams = alpha_bigrams.tolist()
	alpha_bigrams2 = alpha_bigrams2.tolist()
	final_alpha_bigrams = set(alpha_bigrams + alpha_bigrams2)
	
	nonalpha_bigrams = binomials_data['Word2'] + ' and' #need to get all the 'bread and' and 'and butter'
	nonalpha_bigrams2 = 'and ' + binomials_data['Word1']
	
	nonalpha_bigrams = nonalpha_bigrams.tolist()
	nonalpha_bigrams2 = nonalpha_bigrams2.tolist()
	final_nonalpha_bigrams = set(nonalpha_bigrams + nonalpha_bigrams2)
	
	
	bigrams_alpha = ['\t'.join(w.split(' ')) for w in final_alpha_bigrams] #our function takes as its input the trigram tab separated instead of space separated
	bigrams_nonalpha = ['\t'.join(w.split(' ')) for w in final_nonalpha_bigrams] #our function takes as its input the trigram tab separated instead of space separated
	bigrams = bigrams_alpha + bigrams_nonalpha
	bigram_frequencies = trigrams_search_parallel(ngrams = bigrams, ngram_type = 'twogram', num_workers=num_workers)

	with open('olmo_bigram_freqs.csv', 'w') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(['ngram', 'count'])
		for ngram, frequency in trigram_frequencies.items():
			writer.writerow([ngram, frequency])
			



#I think the rest can get done in R more easily

def main():
        print("Calculating onegram corpus size...")
        corpus_size = get_corpus_size()
        print(corpus_size)
        print("Calculating count of 'and'")
        count_and = trigrams_search_parallel(ngrams = ['and'], ngram_type = 'onegram')
        print(count_and)
        print("Getting bigram counts now...")
        get_bigram_counts()
    
    

main()
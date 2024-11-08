import pandas as pd
import glob
import concurrent.futures
from functools import partial
import time
import logging
from tqdm import tqdm  # Import tqdm for the progress bar
from collections import defaultdict
import csv 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_file_for_trigrams(file, trigrams, chunk_size=40000000):
    # Initialize a dictionary to hold word frequencies for each trigram
    word_freqs = defaultdict(int)
    
    # Read the file in chunks
    for chunk in pd.read_csv(file, compression='gzip', encoding='utf-8', chunksize=chunk_size, usecols=['ngram', 'count']):
        # Filter rows where 'ngram' is in the list of trigrams
        filtered_chunk = chunk[chunk['ngram'].isin(trigrams)]
        
        # Sum the counts for each trigram in the chunk
        for trigram, group in filtered_chunk.groupby('ngram'):
            word_freqs[trigram] += group['count'].sum()
    
    return word_freqs, file  # Return the dictionary of frequencies and the file name

def trigrams_search_parallel(ngrams, ngram_type = 'threegram', chunk_size=10000000, num_workers=None): #ngram_type can be onegram, twogram, or threegram
    
    ngram_dirs = {
        'onegram': 'onegram_files',
        'twogram': 'twogram_files',
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


def main(ngram_type):
    binomials_data = pd.read_csv('binomials.csv')
    num_workers = 60
    if ngram_type == 'twogram' or ngram_type == 'threegram':
        alpha_trigrams = binomials_data['Alpha'].tolist()
        nonalpha_trigrams = binomials_data['Nonalpha'].tolist()
        #trigrams = ["the significance of", "the dog is", "my favorite animal"]  # Test trigrams
        trigrams_alpha = ['\t'.join(w.split(' ')) for w in alpha_trigrams] #our function takes as its input the trigram tab separated instead of space separated
        trigrams_nonalpha = ['\t'.join(w.split(' ')) for w in nonalpha_trigrams] #our function takes as its input the trigram tab separated instead of space separated
        trigrams = trigrams_alpha + trigrams_nonalpha
        trigram_frequencies = trigrams_search_parallel(ngrams = trigrams, ngram_type = ngram_type, num_workers=num_workers)
        
        with open('olmo_binomial_freqs.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['ngram', 'count'])
            for ngram, frequency in trigram_frequencies.items():
                writer.writerow([trigram, frequency])
    else:
        
        word1 = binomials_data['Word1'].tolist()
        word2 = binomials_data['Word2'].tolist()
        ngrams = word1 + word2
        ngram_frequencies = trigrams_search_parallel(ngrams, ngram_type = ngram_type, num_workers = num_workers)
        
        with open('olmo_onegram_freqs.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['ngram', 'count'])
            for ngram, frequency in ngram_frequencies.items():
                writer.writerow([ngram, frequency])
    
    
    


if __name__ == "__main__":
    t1 = time.perf_counter() 
    main(ngram_type='onegram')
    t2 = time.perf_counter()
    print(f"Total time taken: {t2 - t1:.2f} seconds")
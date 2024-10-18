#check integrity of the database
from os import listdir
import gzip
from os.path import isfile, join
#Should parallelize this in the future

def check_integrity(gzip_file):
	chunksize=10000000 # 10 Mbytes
	ok = True
	with gzip.open(gzip_file, 'rb') as f:
		try:
			while f.read(chunksize) != b'':
				pass
		# the file is not a gzip file.
		except gzip.BadGzipFile:
			ok = False
		# EOFError: Compressed file ended before the end-of-stream marker was reached
		# a truncated gzip file.
		except EOFError:
			ok = False
    return ok
			
def list_of_bad_files():
	bad_files = []
    path = 'Dolma/'
	gzip_files = gzip_files = [(path + f) for f in listdir(path) if isfile(join(path, f))]	#all the gzip files in the directory
    num_files = len([name for name in os.listdir(path) if os.path.isfile(name)])
	for i,file in enumerate(gzip_files):
        print(f"Progress: {i} / {num_files}.")
		if not check_integrity(file):
			bad_files.append(file)
            print(file)
    return bad_files 
            
            
list_of_bad_files()

def list_of_bad_files():
        bad_files = []
        path = 'Dolma/'
        gzip_files = [(path + f) for f in listdir(path) if isfile(join(path, f))]
        num_files = len([name for name in os.listdir(path)])
        for i,file in enumerate(gzip_files):
            print(f"Progress: {i} / {num_files}.")
            if not check_integrity(file):
                bad_files.append(file)
                print(file)
        return bad_files
        

files_to_redownload = list_of_bad_files()
print(files_to_redownload)
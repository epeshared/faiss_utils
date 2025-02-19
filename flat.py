import subprocess
import numpy as np
import os
import time
import faiss

# Define dimensions and query sizes to test
dimensions = [256, 1024]  # List of dimensions to test
query_sizes = [1, 16, 32, 64, 128]  # Number of queries to test
nb = 10000000  # Database size
nb_mbytes = nb/1000/1000;  # Size of the database in MB

# FAISS parameters
nlist = 100
m = 8
k = 4

# Check if EMON is enabled
emon_enabled = os.getenv("EMON_ENABLE", "0") == "1"

print("data size: ", nb_mbytes, "M")

# Loop through each dimension and query size
test_results = []  # To store results for each dimension and query size
for d in dimensions:
    for nq in query_sizes:
        print(f"Testing with dimension: {d}, number of queries: {nq}")

        # Generate random data
        np.random.seed(1234)

        xq = np.random.random((nq, d)).astype('float32')
        xq[:, 0] += np.arange(nq) / 1000.

        # Define file name for the index
        index_file = f"trained_index_d{d}_db{nb_mbytes}M.flat"

        # Check if index already exists
        if os.path.exists(index_file):
            print(f"Loading index from {index_file}...")
            index = faiss.read_index(index_file)
        else:
            print("Training new index...")
            index = faiss.IndexFlatIP(d)
            # index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
            xb = np.random.random((nb, d)).astype('float32')
            xb[:, 0] += np.arange(nb) / 1000.            
            index.train(xb)
            index.add(xb)
            faiss.write_index(index, index_file)
            print(f"Index saved to {index_file}.")

        # Sanity check
        # D, I = index.search(xb[:5], k)
        # print("Sanity check results:")
        # print("Indices:", I)
        # print("Distances:", D)

        # Set nprobe for better accuracy
        index.nprobe = 10

        loops = 1
        for i in range(loops):
            index.search(xq, k)        

        # Performance test
        # print("Starting search performance test...")
        loops = 5       
        
        if emon_enabled:
            print("kickoff emon...")
            subprocess.Popen("emon -collect-edp > emon.dat &", shell=True)        

        t0 = time.time()
        for i in range(loops):
            index.search(xq, k)
        search_time = (time.time() - t0) / loops

        if emon_enabled:
            print("stop emon...")
            subprocess.run("emon -stop", shell=True)
            subprocess.run(f"mv emon.dat emon_gnr_flat_d{d}_db{nb_mbytes}M_q{nq}_avx512_sncon_c6off_1s43c.dat", shell=True)        
        print(f"Avg. search time for dimension {d}, number of queries {nq}: {search_time:.6f} s")

        # Store results
        test_results.append((d, nq, search_time))

# Print summary results
print("\nSummary of Results:")
for d, nq, t in test_results:
    print(f"Dimension: {d}, Number of Queries: {nq}, Avg. Search Time: {t:.6f} s")

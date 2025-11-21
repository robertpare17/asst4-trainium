Part 1:

1.1 The execution time for running the tiled vector add kernel with chunk size 1 and vector size 25,600 was 27404 microseconds
1.2 The execution time for running the tiled vector add kernel with chunk size 128 and vector size 25,600 was 206 microseconds. This is 135x faster
because the NKI compiler can load `ROW_CHUNK` elements in parallel from HBM and then perform `ROW_CHUNK` wide vector add on the vectors on chip in SBUF. 
1.3 When running with `ROW_CHUNK` equal to 256, we get an error because we exceed the architecture limitation of having the partition dimension being at most 128. 

2a.1 The execution time for running the stream vector add kernel with free dimension 2, parition dimension 128 and vector size 25,600 was 53 microseconds. 
This about 4x faster than compared to `vector_add_tiled` with `ROW_CHUNK` equal to 128.
2a.2 I chose a `FREE_DIM` value of 200 since this is largest possible free dimension for a vector of size 25,600 (128 x 200 = 25,600). Using `FREE_DIM` equal to 200 reduced the execution time to 19 microseconds. This is about 2.5x faster than using `FREE_DIM` equal to 2 and 10x faster than `vector_add_tiled` with `ROW_CHUNK` equal to 128. 

2b.1 When `FREE_DIM` equals 2000, the total *dma_transfer_count* was 3 and the kernel exeuction time was 3.4e-05. When `FREE_DIM` equals 1000, the total *dma_transfer_count* was 6 and the kernel exeuction time was 2.9e-05. 

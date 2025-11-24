Part 1:

1.1 The execution time for running the tiled vector add kernel with chunk size 1 and vector size 25,600 was 27404 microseconds
1.2 The execution time for running the tiled vector add kernel with chunk size 128 and vector size 25,600 was 206 microseconds. This is 135x faster
because the NKI compiler can load `ROW_CHUNK` elements in parallel from HBM and then perform `ROW_CHUNK` wide vector add on the vectors on chip in SBUF. 
1.3 When running with `ROW_CHUNK` equal to 256, we get an error because we exceed the architecture limitation of having the partition dimension being at most 128. 

2a.1 The execution time for running the stream vector add kernel with free dimension 2, parition dimension 128 and vector size 25,600 was 53 microseconds. 
This about 4x faster than compared to `vector_add_tiled` with `ROW_CHUNK` equal to 128.

2a.2 I chose a `FREE_DIM` value of 200 since this is largest possible free dimension for a vector of size 25,600 (128 x 200 = 25,600). Using `FREE_DIM` equal to 200 reduced the execution time to 19 microseconds. This is about 2.5x faster than using `FREE_DIM` equal to 2 and 10x faster than `vector_add_tiled` with `ROW_CHUNK` equal to 128. 

2b.2 When `FREE_DIM` equals 2000, the total *dma_transfer_count* was 3 and the kernel exeuction time was 3.4e-05. When `FREE_DIM` equals 1000, the total *dma_transfer_count* was 6 and the kernel exeuction time was 2.9e-05. 

2b.3 Despite `FREE_DIM = 1000` requiring more DMA transfers, it's faster because of better pipelining. With a smaller `FREE_DIM` size, the Trainium chip can overlap data transfer with computation by the Vector Engine, keeping both DMA engines and compute units busy simulataneously. This results in higher overall throughput. 

3.1 The execution time for `vector_add_direct_allocation` with a vector of size 256,000 was 29 microseconds. The execution time for `vector_add_stream` with `FREE_DIM=1000` was 30 microseconds. They are the same because `vector_add_direct_allocation` manually performs the allocations that the Neuron compiler automatically performs for `vector_add_stream`.

3.2 There are 4 physical tiles allocated for each tensor. If we set the number of physical tiles to be too large then we could potentially run out of memory in SBUF or be wasting valuable SBUF memory which could be used for other operations.

3.3 If we don't have different offsets for each tensor, then we will corrupt the data and hence the correctness of the operation. The result of `vector_add_direct_allocation` on a vector of size 256,000 is a failed correctness test. 

import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    TILE_K = nl.tile_size.pmax # 128
    TILE_M = nl.tile_size.gemm_stationary_fmax # 128
    TILE_N = nl.tile_size.gemm_moving_fmax # 512

    n_tiles_c_in = in_channels // TILE_K
    n_tiles_c_out = out_channels // TILE_M
    out_spatial = out_height * out_width
    n_tiles_n = (out_spatial + TILE_N - 1) // TILE_N

    X_reshaped = X.reshape((batch_size, in_channels, input_height * input_width))
    W_reshaped = W.reshape((out_channels, in_channels, filter_height * filter_width))

    # Define tile indices
    i_w = nl.mgrid[0:TILE_M, 0:TILE_K]
    i_x = nl.mgrid[0:TILE_K, 0:TILE_N]
    i_res = nl.mgrid[0:TILE_M, 0:TILE_N]

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # raise RuntimeError("Please fill your implementation of computing convolution"
        #                    " of X[b] with the weights W and bias b, followed by a"
        #                    " maxpool and store the result in X_out[b]")

        X_flat = X_reshaped[b, :, :]

        # Can only allocate one tile of output in SBUF at a time since out_channels may be > 128
        # Allocate full output in HBM for now until optimizing for tiled approach in SBUF
        # conv_out = nl.zeros(
        #     shape=(out_channels, out_height * out_width),
        #     dtype=X.dtype,
        #     buffer=nl.sbuf,
        # )
        conv_out = nl.zeros(
            shape=(n_tiles_c_out, nl.par_dim(TILE_M), out_height * out_width),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )

        # Iterate over filter positions (i, j)
        for i in nl.sequential_range(filter_height):
            for j in nl.sequential_range(filter_width):
                filter_index = i * filter_width + j

                # Create shifted input aligned with filter position
                # For each output position (oh, ow), input position is (oh + i, ow + j)
                # Note: same issue as with out_channels -- in_channels may be > 128
                # X_shifted = nl.ndarray(
                #     shape=(in_channels, out_height * out_width),
                #     dtype=X.dtype,
                #     buffer=nl.sbuf,
                # )
                X_shifted_tiles = nl.ndarray(
                    shape=(n_tiles_c_in, nl.par_dim(TILE_K), out_height * out_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )

                # Load all input channel tiles for this filter position
                for k_tile in nl.affine_range(n_tiles_c_in):
                    k_start = k_tile * TILE_K
                    k_end = (k_tile + 1) * TILE_K

                    for oh in nl.affine_range(out_height):
                        for ow in nl.affine_range(out_width):
                            ih = oh + i
                            iw = ow + j
                            in_idx = ih * input_width + iw
                            out_idx = oh * out_width + ow
                            # load one channel tile at a time
                            X_shifted_tiles[k_tile, :, out_idx] = nl.load(X_flat[k_start:k_end, in_idx])

                # Now perform tiled matrix multiplication
                # Iterate over output channel tiles
                for m_tile in nl.affine_range(n_tiles_c_out):
                    m_start = m_tile * TILE_M
                    # m_end = min((m_tile + 1) * TILE_M, out_channels)
                    m_end = (m_tile + 1) * TILE_M

                    # Load weight tiles for this output channel tile across all input channels
                    # Shape: (n_tiles_c_in, TILE_M, TILE_K)
                    W_tiles = nl.ndarray(
                        shape=(n_tiles_c_in, nl.par_dim(TILE_M), TILE_K, filter_height * filter_width),
                        dtype=W.dtype,
                        buffer=nl.sbuf,
                    )

                    for k_tile in nl.affine_range(n_tiles_c_in):
                        k_start = k_tile * TILE_K
                        k_end = (k_tile + 1) * TILE_K

                        # Load weight tile: out_channels[m_start:m_end] x in_channels[k_start:k_end]
                        W_tiles[k_tile, :, :, :] = nl.load(
                            W_reshaped[m_start:m_end, k_start:k_end, :]
                        )

                    for n_tile in nl.affine_range(n_tiles_n):
                        n_start = n_tile * TILE_N
                        n_end = min((n_tile + 1) * TILE_N, out_spatial)
                        n_size = n_end - n_start

                        res_psum = nl.zeros(
                            shape=(TILE_M, TILE_N),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )

                        for k_tile in nl.affine_range(n_tiles_c_in):

                            i_rhs_f = nl.arange(TILE_N)[None, :]
                            res_psum += nl.matmul(
                                W_tiles[k_tile, i_w.p, i_w.x, filter_index],
                                X_shifted_tiles[k_tile, i_x.p, i_x.x][i_rhs_f < n_size],
                            )
                            # k_start = k_tile * TILE_K
                            # k_end = (k_tile + 1) * TILE_K

                            # filter_index = i * filter_width + j

                            # w_tile = nl.ndarray((TILE_M, TILE_K, filter_height * filter_width), dtype=W.dtype, buffer=nl.sbuf)
                            # w_tile[:m_size, :, :] = nl.load(W_reshaped[m_start:m_end, k_start:k_end, :])

                            # x_tile = nl.ndarray((TILE_K, TILE_N), dtype=X.dtype, buffer=nl.sbuf)
                            # x_tile[:, :n_size] = X_shifted[k_start:k_end, n_start:n_end]

                            # res_psum[:m_size, :n_size] += nl.matmul(w_tile[:m_size, :, filter_index], x_tile[:, :n_size])

                        result_sbuf = nl.copy(res_psum, dtype=X.dtype)
                        conv_out[m_tile, i_res.p, n_start + i_res.x] = nl.add(
                            conv_out[m_tile, i_res.p, n_start + i_res.x],
                            result_sbuf[i_res.p, i_res.x],
                            mask=(n_start + i_res.x < out_spatial)
                        )



        # Worry about bias and maxpooling later...
        # Maxpooling - work directly with tiled conv_out format
        if pool_size == 1:
            # No pooling - stitch tiles directly to output
            for m_tile in nl.affine_range(n_tiles_c_out):
                m_start = m_tile * TILE_M
                m_end = (m_tile + 1) * TILE_M
                
                # Reshape this tile's spatial dimension
                conv_out_tile_flat = nl.ndarray(
                    shape=(TILE_M, out_height * out_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                conv_out_tile_flat[:, :] = conv_out[m_tile, :, :]
                conv_out_tile_spatial = conv_out_tile_flat.reshape((TILE_M, out_height, out_width))
                
                # Store directly to HBM
                nl.store(X_out[b, m_start:m_end, :, :], value=conv_out_tile_spatial)
        else:
            # Maxpooling with tiled format
            # Process each output channel tile separately
            for m_tile in nl.affine_range(n_tiles_c_out):
                m_start = m_tile * TILE_M
                m_end = (m_tile + 1) * TILE_M
                
                # Reshape this tile's spatial dimension
                conv_out_tile_flat = nl.ndarray(
                    shape=(TILE_M, out_height * out_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                conv_out_tile_flat[:, :] = conv_out[m_tile, :, :]
                conv_out_tile_spatial = conv_out_tile_flat.reshape((TILE_M, out_height, out_width))
                
                # Perform maxpooling on this tile
                pooled_tile = nl.ndarray(
                    shape=(TILE_M, out_pool_height, out_pool_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                
                for c_local in nl.affine_range(TILE_M):
                    for ph in nl.affine_range(out_pool_height):
                        for pw in nl.affine_range(out_pool_width):
                            h0, w0 = ph * 2, pw * 2
                            max_val = nl.max(
                                nl.max(conv_out_tile_spatial[c_local, h0, w0], 
                                      conv_out_tile_spatial[c_local, h0, w0+1]),
                                nl.max(conv_out_tile_spatial[c_local, h0+1, w0], 
                                      conv_out_tile_spatial[c_local, h0+1, w0+1])
                            )
                            pooled_tile[c_local, ph, pw] = max_val
                
                # Store pooled tile to HBM
                nl.store(X_out[b, m_start:m_end, :, :], value=pooled_tile)

    return X_out

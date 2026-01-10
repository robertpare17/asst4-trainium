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

    # Calculate maximum spatial tile
    MAX_FREE_DIM_BYTES = 192 * 1024  # 192KB
    dtype_bytes = 4 if X.dtype == nl.float32 else 2

    MAX_SPATIAL_ELEMENTS = MAX_FREE_DIM_BYTES // (n_tiles_c_out * dtype_bytes)
    SPATIAL_TILE = min(TILE_N, MAX_SPATIAL_ELEMENTS)

    n_tiles_n = (out_spatial + SPATIAL_TILE - 1) // SPATIAL_TILE

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

        # Initialize conv_out buffer to hold convolution results before pooling
        conv_output_hbm = nl.ndarray(
            shape=(out_channels, out_height, out_width),
            dtype=X.dtype,
            buffer=nl.hbm,
        )

        for m_tile in nl.affine_range(n_tiles_c_out):
            m_start = m_tile * TILE_M
            m_end = (m_tile + 1) * TILE_M

            for n_tile in nl.affine_range(n_tiles_n):
                n_start = n_tile * SPATIAL_TILE
                n_end = min((n_tile + 1) * SPATIAL_TILE, out_spatial)
                n_size = n_end - n_start

                # Allocate accumulator for this output block in SBUF
                conv_out_block = nl.zeros(
                    shape=(TILE_M, SPATIAL_TILE),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )

                # Define tile indices for this spatial tile size
                i_w = nl.mgrid[0:TILE_M, 0:TILE_K]
                i_x = nl.mgrid[0:TILE_K, 0:SPATIAL_TILE]
                i_block = nl.mgrid[0:TILE_M, 0:SPATIAL_TILE]

                # Accumulate over all filter positions and input channels
                for i in nl.sequential_range(filter_height):
                    for j in nl.sequential_range(filter_width):
                        filter_index = i * filter_width + j

                        # Load weight tiles for this output channel
                        # Shape: (TILE_M, TILE_K, n_tiles_c_in, filter_height * filter_width)
                        W_tiles = nl.ndarray(
                            shape=(TILE_M, TILE_K, n_tiles_c_in, filter_height * filter_width),
                            dtype=W.dtype,
                            buffer=nl.sbuf,
                        )

                        for k_tile in nl.affine_range(n_tiles_c_in):
                            k_start = k_tile * TILE_K
                            k_end = (k_tile + 1) * TILE_K

                            # Load weight tile: out_channels[m_start:m_end] x in_channels[k_start:k_end]
                            W_tiles[:, :, k_tile, :] = nl.load(
                                W_reshaped[m_start:m_end, k_start:k_end, :]
                            )

                        res_psum = nl.zeros(
                            shape=(TILE_M, SPATIAL_TILE),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )

                        # Process this spatial block across all input channel tiles
                        for k_tile in nl.affine_range(n_tiles_c_in):
                            k_start = k_tile * TILE_K
                            k_end = (k_tile + 1) * TILE_K

                            # Load shifted for this k_tile and spatial region
                            X_shifted_tile = nl.ndarray(
                                shape=(TILE_K, SPATIAL_TILE),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )

                            # NOTE: This is causing an off by one index error. Should get to the bottom of this.
                            # Load input data for this spatial block
                            for spatial_idx in nl.affine_range(SPATIAL_TILE):
                                global_spatial_idx = n_start + spatial_idx
                                oh = global_spatial_idx // out_width
                                ow = global_spatial_idx % out_width
                                ih = oh + i
                                iw = ow + j
                                in_idx = ih * input_width + iw
                                X_shifted_tile[:, spatial_idx] = nl.load(
                                    X_flat[k_start:k_end, in_idx],
                                    mask=(spatial_idx < n_size)
                                )
                            
                            # Use operand masking for boundary handling
                            i_rhs_f = nl.arange(SPATIAL_TILE)[None, :]
                            res_psum += nl.matmul(
                                W_tiles[i_w.p, i_w.x, k_tile, filter_index],
                                X_shifted_tile[i_x.p, i_x.x][i_rhs_f < n_size],
                            )

                        # Accumulate into conv_out_block
                        result_sbuf = nl.copy(res_psum, dtype=X.dtype)
                        i_block = nl.mgrid[0:TILE_M, 0:SPATIAL_TILE]
                        conv_out_block[i_block.p, i_block.x] = nl.add(
                            conv_out_block[i_block.p, i_block.x],
                            result_sbuf[i_block.p, i_block.x],
                            mask=(i_block.x < n_size)
                        )

                # After accumulating over all input channels and filter positions,
                # Store this spaatial block to temporary HBM tensor
                for local_idx in nl.affine_range(n_size):
                    global_idx = n_start + local_idx
                    out_h = global_idx // out_width
                    out_w = global_idx % out_width
                    nl.store(
                        conv_output_hbm[m_start:m_end, out_h, out_w],
                        value=conv_out_block[:, local_idx],
                    )

        # After all convolution is done, copy to output (with pooling if needed)
        if pool_size == 1:
            # No pooling - just copy conv output to final output
            for m_tile in nl.affine_range(n_tiles_c_out):
                m_start = m_tile * TILE_M
                m_end = (m_tile + 1) * TILE_M
                for h in nl.affine_range(out_height):
                    for w in nl.affine_range(out_width):
                        val = nl.load(conv_output_hbm[m_start:m_end, h, w])
                        nl.store(X_out[b, m_start:m_end, h, w], value=val)
        else:  # pool_size == 2
            # Perform maxpooling: load 2x2 regions, compute max, store
            for m_tile in nl.affine_range(n_tiles_c_out):
                m_start = m_tile * TILE_M
                m_end = (m_tile + 1) * TILE_M
                
                for ph in nl.affine_range(out_pool_height):
                    for pw in nl.affine_range(out_pool_width):
                        h0, w0 = ph * 2, pw * 2
                        
                        # Load the 2x2 region for all channels in this tile
                        vals = nl.ndarray(shape=(TILE_M, 4), dtype=X.dtype, buffer=nl.sbuf)
                        vals[:, 0] = nl.load(conv_output_hbm[m_start:m_end, h0, w0])
                        vals[:, 1] = nl.load(conv_output_hbm[m_start:m_end, h0, w0+1])
                        vals[:, 2] = nl.load(conv_output_hbm[m_start:m_end, h0+1, w0])
                        vals[:, 3] = nl.load(conv_output_hbm[m_start:m_end, h0+1, w0+1])
                        
                        # Compute max across the 2x2 region
                        max_val = nl.max(nl.max(vals[:, 0], vals[:, 1]), 
                                        nl.max(vals[:, 2], vals[:, 3]))
                        
                        nl.store(X_out[b, m_start:m_end, ph, pw], value=max_val)

    return X_out

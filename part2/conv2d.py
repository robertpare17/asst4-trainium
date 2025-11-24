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
    TILE_M = nl.tile_size.pmax # 128
    TILE_N = nl.tile_size.gemm_moving_fmax # 512

    n_tiles_c_in = in_channels // TILE_K
    n_tiles_c_out = out_channels // TILE_M
    out_spatial = out_height * out_width
    n_tiles_n = (out_spatial + TILE_N - 1) // TILE_N

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # raise RuntimeError("Please fill your implementation of computing convolution"
        #                    " of X[b] with the weights W and bias b, followed by a"
        #                    " maxpool and store the result in X_out[b]")

        X_img = X[b, :, :, :]

        X_flat = X_img.reshape((in_channels, input_height * input_width))

        conv_out = nl.zeros(
            shape=(out_channels, out_height, out_width),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )

        # Iterate over filter positions (i, j)
        for i in nl.affine_range(filter_height):
            for j in nl.affine_range(filter_width):
                # Get filter slice at (i,j): shape (out_channels, in_channels)
                W_slice = W[: , :, i, j]

                # Create shifted input aligned with filter position
                # For each output position (oh, ow), input position is (oh + i, ow + j)
                X_shifted = nl.ndarray(
                    shape=(in_channels, out_height * out_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )

                for oh in nl.affine_range(out_height):
                    for ow in nl.affine_range(out_width):
                        ih = oh + i
                        iw = ow + j
                        in_idx = ih * input_width + iw
                        out_idx = oh * out_width + ow
                        X_shifted[:, out_idx] = nl.copy(X_flat[:, in_idx])

                # Matrix multiply: W_slice @ X_shifted
                # Shape: (out_channels, in_channels) @ (in_channels, out_height*out_width)
                # Result: (out_channels, out_height*out_width)

                for m_tile in nl.affine_range(n_tiles_c_out):
                    for n_tile in nl.affine_range(n_tiles_n):
                        n_start = n_tile * TILE_N
                        n_end = min((n_tile + 1) * TILE_N, out_spatial)
                        n_size = n_end - n_start

                        psum = nl.zeros(
                            shape=(TILE_M, TILE_N),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )

                        for k_tile in nl.affine_range(n_tiles_c_in):
                            w_tile = nl.ndarray((TILE_M, TILE_K), dtype=W.dtype, buffer=nl.sbuf)
                            w_tile[...] = nl.load(
                                W_slice[
                                    m_tile * TILE_M : (m_tile + 1) * TILE_M,
                                    k_tile * TILE_K : (k_tile + 1) * TILE_K,
                                ]
                            )

                            x_tile = nl.ndarray((TILE_K, TILE_N), dtype=X.dtype, buffer=nl.sbuf)
                            x_tile[...] = nl.load(
                                X_shifted[
                                    k_tile * TILE_K : (k_tile + 1) * TILE_K,
                                    n_start : n_end,
                                ]
                                mask=(x_tile[:, :n_size] == 1)
                            )

                            psum[:, :n_size] += nl.matmul(w_tile, x_tile[:, :n_size])

                        result = nl.copy(psum, dtype=X.dtype)

                        for m_idx in nl.affine_range(TILE_M):
                            for n_idx in nl.affine_range(n_size):
                                global_m = m_tile * TILE_M + m_idx
                                global_n = n_start + n_idx
                                oh = global_n // out_width
                                ow = global_n % out_width
                                conv_out[global_m, oh, ow] += result[m_idx, n_idx]

        # Add bias
        for c in nl.affine_range(out_channels):
            conv_out[c, :, :] += bias[c]

        # Maxpooling
        if pool_size == 1:
            nl.store(X_out[b, :, :, :], value=conv_out)
        else:
            pooled = nl.ndarray(
                shape=(out_channels, out_pool_height, out_pool_width),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )

            for c in nl.affine_range(out_channels):
                for ph in nl.affine_range(out_pool_height):
                    for pw in nl.affine_range(out_pool_width):
                        h0, w0 = ph * 2, pw * 2
                        max_val = nl.max(
                            nl.max(conv_out[c, h0, w0], conv_out[c, h0, w0+1]),
                            nl.max(conv_out[c, h0+1, w0], conv_out[c, h0+1, w0+1])
                        )
                        pooled[c, ph, pw] = max_val
            
            nl.store(X_out[b, :, :, :], value=pooled)

    return X_out

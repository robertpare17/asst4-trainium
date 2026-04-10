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
    out_width  = input_width  - filter_width  + 1

    out_pool_height = out_height // pool_size
    out_pool_width  = out_width  // pool_size

    assert in_channels % 128 == 0
    assert out_channels % 128 == 0
    assert nl.tile_size.gemm_moving_fmax >= out_width

    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    TILE_K = nl.tile_size.pmax                  # 128
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax      # 512

    n_tiles_c_in  = in_channels  // TILE_K
    n_tiles_c_out = out_channels // TILE_M

    # Compile-time spatial tiling.
    if out_height * out_width <= TILE_N:
        ROW_TILE = out_height
    else:
        ROW_TILE = TILE_N // out_width
        if out_height % ROW_TILE != 0:
            ROW_TILE = 1

    N_SPATIAL   = ROW_TILE * out_width
    n_row_tiles = out_height // ROW_TILE
    STRIP_ROWS  = ROW_TILE + filter_height - 1

    X_reshaped = X.reshape((batch_size, in_channels, input_height * input_width))
    W_reshaped = W.reshape((out_channels, in_channels, filter_height * filter_width))

    i_w   = nl.mgrid[0:TILE_M, 0:TILE_K]
    i_x   = nl.mgrid[0:TILE_K, 0:N_SPATIAL]
    i_row = nl.mgrid[0:TILE_M, 0:out_width]

    # ================================================================
    # TOP-LEVEL weight preload (outside all affine_range loops).
    # Python range() here gives plain Python ints, not LoopVars.
    # W and bias don't depend on the batch index, so loading once is correct.
    # For n_tiles_c_out=2 (the benchmark case) both wt_0 and wt_1 are loaded
    # into SBUF and reused across every row_tile and every batch element.
    # ================================================================
    wt_0 = nl.ndarray(
        shape=(TILE_M, TILE_K, n_tiles_c_in, filter_height * filter_width),
        dtype=W.dtype,
        buffer=nl.sbuf,
    )
    for k_tile in nl.affine_range(n_tiles_c_in):
        k_s = k_tile * TILE_K
        wt_0[:, :, k_tile, :] = nl.load(W_reshaped[0:TILE_M, k_s:k_s + TILE_K, :])
    bt_0 = nl.load(bias[0:TILE_M])

    if n_tiles_c_out >= 2:
        wt_1 = nl.ndarray(
            shape=(TILE_M, TILE_K, n_tiles_c_in, filter_height * filter_width),
            dtype=W.dtype,
            buffer=nl.sbuf,
        )
        for k_tile in nl.affine_range(n_tiles_c_in):
            k_s = k_tile * TILE_K
            wt_1[:, :, k_tile, :] = nl.load(W_reshaped[TILE_M:2 * TILE_M, k_s:k_s + TILE_K, :])
        bt_1 = nl.load(bias[TILE_M:2 * TILE_M])

    for b in nl.affine_range(batch_size):
        X_flat = X_reshaped[b, :, :]

        if pool_size > 1:
            conv_hbm = nl.ndarray(
                shape=(out_channels, out_height, out_width),
                dtype=X.dtype,
                buffer=nl.hbm,
            )

        for row_tile in nl.affine_range(n_row_tiles):
            oh_start = row_tile * ROW_TILE

            # One PSUM per output-channel tile.  Both are alive simultaneously,
            # accumulating contributions from every (k_tile, i, j).  X_tile is
            # built ONCE per (k_tile, i, j) and shared by both matmuls — halving
            # the SBUF copy work vs the m_tile-outer structure.
            psum_0 = nl.zeros(
                shape=(TILE_M, N_SPATIAL), dtype=nl.float32, buffer=nl.psum
            )
            if n_tiles_c_out >= 2:
                psum_1 = nl.zeros(
                    shape=(TILE_M, N_SPATIAL), dtype=nl.float32, buffer=nl.psum
                )

            for k_tile in nl.affine_range(n_tiles_c_in):
                k_start = k_tile * TILE_K
                k_end   = (k_tile + 1) * TILE_K

                # Load STRIP_ROWS input rows as separate DMA transactions so the
                # compiler can pipeline them with each other and with compute.
                strip = nl.ndarray(
                    shape=(TILE_K, STRIP_ROWS * input_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                for sr in nl.affine_range(STRIP_ROWS):
                    ih = oh_start + sr
                    strip[:, sr * input_width:(sr + 1) * input_width] = nl.load(
                        X_flat[k_start:k_end, ih * input_width:(ih + 1) * input_width]
                    )

                for i in nl.sequential_range(filter_height):
                    for j in nl.sequential_range(filter_width):
                        filter_idx = i * filter_width + j

                        # Build X_tile once; reuse for all m_tiles.
                        X_tile = nl.ndarray(
                            shape=(TILE_K, N_SPATIAL),
                            dtype=X.dtype,
                            buffer=nl.sbuf,
                        )
                        for local_oh in nl.affine_range(ROW_TILE):
                            sc = (local_oh + i) * input_width + j
                            dc = local_oh * out_width
                            X_tile[:, dc:dc + out_width] = strip[:, sc:sc + out_width]

                        psum_0 += nl.matmul(
                            wt_0[i_w.p, i_w.x, k_tile, filter_idx],
                            X_tile[i_x.p, i_x.x],
                        )
                        if n_tiles_c_out >= 2:
                            psum_1 += nl.matmul(
                                wt_1[i_w.p, i_w.x, k_tile, filter_idx],
                                X_tile[i_x.p, i_x.x],
                            )

            # ---- Store results for m_tile = 0 ----
            result_0 = nl.copy(psum_0, dtype=X.dtype)
            for local_oh in nl.affine_range(ROW_TILE):
                src_col   = local_oh * out_width
                global_oh = oh_start + local_oh
                row_buf   = nl.ndarray(
                    shape=(TILE_M, out_width), dtype=X.dtype, buffer=nl.sbuf
                )
                row_buf[i_row.p, i_row.x] = result_0[i_row.p, src_col + i_row.x]
                row_with_bias = nl.add(row_buf, bt_0)
                if pool_size == 1:
                    nl.store(X_out[b, 0:TILE_M, global_oh, :], value=row_with_bias)
                else:
                    nl.store(conv_hbm[0:TILE_M, global_oh, :], value=row_with_bias)

            # ---- Store results for m_tile = 1 ----
            if n_tiles_c_out >= 2:
                result_1 = nl.copy(psum_1, dtype=X.dtype)
                for local_oh in nl.affine_range(ROW_TILE):
                    src_col   = local_oh * out_width
                    global_oh = oh_start + local_oh
                    row_buf   = nl.ndarray(
                        shape=(TILE_M, out_width), dtype=X.dtype, buffer=nl.sbuf
                    )
                    row_buf[i_row.p, i_row.x] = result_1[i_row.p, src_col + i_row.x]
                    row_with_bias = nl.add(row_buf, bt_1)
                    if pool_size == 1:
                        nl.store(X_out[b, TILE_M:2 * TILE_M, global_oh, :], value=row_with_bias)
                    else:
                        nl.store(conv_hbm[TILE_M:2 * TILE_M, global_oh, :], value=row_with_bias)

        # ---- MaxPool phase (pool_size > 1 only) ----
        if pool_size > 1:
            i_pw = nl.mgrid[0:TILE_M, 0:out_pool_width]

            for m_tile in nl.affine_range(n_tiles_c_out):
                m_start = m_tile * TILE_M
                m_end   = (m_tile + 1) * TILE_M

                for ph in nl.affine_range(out_pool_height):
                    h0 = ph * 2
                    h1 = ph * 2 + 1

                    row0 = nl.load(conv_hbm[m_start:m_end, h0, :])
                    row1 = nl.load(conv_hbm[m_start:m_end, h1, :])

                    row_max = nl.maximum(row0, row1)

                    even_cols = nl.ndarray(shape=(TILE_M, out_pool_width), dtype=X.dtype, buffer=nl.sbuf)
                    odd_cols  = nl.ndarray(shape=(TILE_M, out_pool_width), dtype=X.dtype, buffer=nl.sbuf)
                    for pw in nl.affine_range(out_pool_width):
                        even_cols[i_pw.p, pw] = row_max[i_pw.p, pw * 2]
                        odd_cols[i_pw.p, pw]  = row_max[i_pw.p, pw * 2 + 1]

                    pooled = nl.maximum(even_cols, odd_cols)
                    nl.store(X_out[b, m_start:m_end, ph, :], value=pooled)

    return X_out

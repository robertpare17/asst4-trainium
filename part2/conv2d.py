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

    # ---------- Compute ROW_TILE at Python (compile) time ----------
    # We want ROW_TILE rows per spatial tile so N_SPATIAL = ROW_TILE * out_width <= TILE_N.
    # We also need out_height % ROW_TILE == 0 to avoid partial last tiles
    # (the bounds checker cannot track ranges derived from Python min() on NKI runtime vars).
    #
    # Case 1: all output rows fit in one TILE_N chunk — process everything in a single tile.
    # Case 2: otherwise, use ROW_TILE = TILE_N // out_width and verify divisibility.
    # Fallback: ROW_TILE = 1 (always safe, one row per matmul).
    if out_height * out_width <= TILE_N:
        ROW_TILE = out_height                   # single-tile case
    else:
        ROW_TILE = TILE_N // out_width          # multi-tile case
        if out_height % ROW_TILE != 0:
            ROW_TILE = 1                        # safe fallback

    N_SPATIAL   = ROW_TILE * out_width          # compile-time constant, <= TILE_N
    n_row_tiles = out_height // ROW_TILE        # exact division — no partial last tile

    X_reshaped = X.reshape((batch_size, in_channels, input_height * input_width))
    W_reshaped = W.reshape((out_channels, in_channels, filter_height * filter_width))

    # mgrid for matmul operands; sizes are compile-time constants.
    i_w   = nl.mgrid[0:TILE_M, 0:TILE_K]
    i_x   = nl.mgrid[0:TILE_K, 0:N_SPATIAL]
    # mgrid for per-row extraction from the N_SPATIAL result.
    i_row = nl.mgrid[0:TILE_M, 0:out_width]

    for b in nl.affine_range(batch_size):
        X_flat = X_reshaped[b, :, :]

        # For pool_size > 1 we need an intermediate HBM buffer because we must
        # read 2×2 neighbourhoods that may span two separately-computed row-tiles.
        # For pool_size == 1 we write directly to X_out and skip this buffer.
        if pool_size > 1:
            conv_hbm = nl.ndarray(
                shape=(out_channels, out_height, out_width),
                dtype=X.dtype,
                buffer=nl.hbm,
            )

        for m_tile in nl.affine_range(n_tiles_c_out):
            m_start = m_tile * TILE_M
            m_end   = (m_tile + 1) * TILE_M

            # Bias for this output-channel block: shape (TILE_M, 1) in NKI.
            bias_tile = nl.load(bias[m_start:m_end])

            # Preload all weights for this output-channel block once; keep in SBUF
            # across all row_tile iterations to avoid redundant HBM reads.
            W_tiles = nl.ndarray(
                shape=(TILE_M, TILE_K, n_tiles_c_in, filter_height * filter_width),
                dtype=W.dtype,
                buffer=nl.sbuf,
            )
            for k_tile in nl.affine_range(n_tiles_c_in):
                k_start = k_tile * TILE_K
                k_end   = (k_tile + 1) * TILE_K
                W_tiles[:, :, k_tile, :] = nl.load(
                    W_reshaped[m_start:m_end, k_start:k_end, :]
                )

            for row_tile in nl.affine_range(n_row_tiles):
                oh_start = row_tile * ROW_TILE   # runtime, range [0, out_height-ROW_TILE]

                # Single PSUM accumulator for ROW_TILE output rows × out_width cols.
                psum = nl.zeros(
                    shape=(TILE_M, N_SPATIAL),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )

                for i in nl.sequential_range(filter_height):
                    for j in nl.sequential_range(filter_width):
                        filter_idx = i * filter_width + j

                        for k_tile in nl.affine_range(n_tiles_c_in):
                            k_start = k_tile * TILE_K
                            k_end   = (k_tile + 1) * TILE_K

                            # Build X_tile: (TILE_K, N_SPATIAL) by loading ROW_TILE rows.
                            # local_oh ∈ [0, ROW_TILE-1] (compile-time bound),
                            # so (oh_start + local_oh) ∈ [0, out_height-1] and
                            # ih ∈ [0, input_height-1] — all accesses are in bounds.
                            X_tile = nl.ndarray(
                                shape=(TILE_K, N_SPATIAL),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )
                            for local_oh in nl.affine_range(ROW_TILE):
                                ih        = oh_start + local_oh + i
                                col_start = ih * input_width + j
                                dst_col   = local_oh * out_width
                                X_tile[:, dst_col:dst_col + out_width] = nl.load(
                                    X_flat[k_start:k_end, col_start:col_start + out_width]
                                )

                            psum += nl.matmul(
                                W_tiles[i_w.p, i_w.x, k_tile, filter_idx],
                                X_tile[i_x.p, i_x.x],
                            )

                # Copy PSUM → SBUF for per-row extraction.
                result_sbuf = nl.copy(psum, dtype=X.dtype)

                # Store each of the ROW_TILE output rows (with bias).
                # local_oh ∈ [0, ROW_TILE-1] so src_col = local_oh*out_width ∈
                # [0, N_SPATIAL-out_width], and src_col + out_width - 1 ≤ N_SPATIAL-1.
                for local_oh in nl.affine_range(ROW_TILE):
                    src_col   = local_oh * out_width
                    global_oh = oh_start + local_oh   # ∈ [0, out_height-1]

                    row_buf = nl.ndarray(
                        shape=(TILE_M, out_width),
                        dtype=X.dtype,
                        buffer=nl.sbuf,
                    )
                    row_buf[i_row.p, i_row.x] = result_sbuf[i_row.p, src_col + i_row.x]
                    row_with_bias = nl.add(row_buf, bias_tile)

                    if pool_size == 1:
                        # Direct write: no intermediate buffer needed.
                        nl.store(X_out[b, m_start:m_end, global_oh, :], value=row_with_bias)
                    else:
                        nl.store(conv_hbm[m_start:m_end, global_oh, :], value=row_with_bias)

        # ---- Pooling phase (pool_size > 1 only) ----
        if pool_size > 1:
            for m_tile in nl.affine_range(n_tiles_c_out):
                m_start = m_tile * TILE_M
                m_end   = (m_tile + 1) * TILE_M
                for ph in nl.affine_range(out_pool_height):
                    for pw in nl.affine_range(out_pool_width):
                        h0 = ph * 2
                        w0 = pw * 2
                        v00 = nl.load(conv_hbm[m_start:m_end, h0,     w0    ])
                        v01 = nl.load(conv_hbm[m_start:m_end, h0,     w0 + 1])
                        v10 = nl.load(conv_hbm[m_start:m_end, h0 + 1, w0    ])
                        v11 = nl.load(conv_hbm[m_start:m_end, h0 + 1, w0 + 1])
                        max_val = nl.maximum(nl.maximum(v00, v01), nl.maximum(v10, v11))
                        nl.store(X_out[b, m_start:m_end, ph, pw], value=max_val)

    return X_out

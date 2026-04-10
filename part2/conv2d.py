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
    # ROW_TILE: number of output rows per spatial tile, chosen so
    # N_SPATIAL = ROW_TILE * out_width <= TILE_N and out_height % ROW_TILE == 0.
    if out_height * out_width <= TILE_N:
        ROW_TILE = out_height           # everything fits in one tile
    else:
        ROW_TILE = TILE_N // out_width  # max rows per tile
        if out_height % ROW_TILE != 0:
            ROW_TILE = 1                # safe fallback

    N_SPATIAL   = ROW_TILE * out_width   # compile-time constant, <= TILE_N
    n_row_tiles = out_height // ROW_TILE

    # STRIP_ROWS: input rows needed to cover one output row-tile at all filter offsets.
    STRIP_ROWS = ROW_TILE + filter_height - 1

    X_reshaped = X.reshape((batch_size, in_channels, input_height * input_width))
    W_reshaped = W.reshape((out_channels, in_channels, filter_height * filter_width))

    i_w   = nl.mgrid[0:TILE_M, 0:TILE_K]
    i_x   = nl.mgrid[0:TILE_K, 0:N_SPATIAL]
    i_row = nl.mgrid[0:TILE_M, 0:out_width]

    for b in nl.affine_range(batch_size):
        X_flat = X_reshaped[b, :, :]

        if pool_size > 1:
            conv_hbm = nl.ndarray(
                shape=(out_channels, out_height, out_width),
                dtype=X.dtype,
                buffer=nl.hbm,
            )

        # ---- Preload ALL weights and biases ONCE (shared across all row_tiles) ----
        # Python range (compile-time unrolling) → static Python list indexing later.
        W_tiles_by_m = []
        bias_by_m    = []
        for m in range(n_tiles_c_out):
            m_s = m * TILE_M
            wt = nl.ndarray(
                shape=(TILE_M, TILE_K, n_tiles_c_in, filter_height * filter_width),
                dtype=W.dtype,
                buffer=nl.sbuf,
            )
            for k in range(n_tiles_c_in):
                k_s = k * TILE_K
                wt[:, :, k, :] = nl.load(W_reshaped[m_s:m_s + TILE_M, k_s:k_s + TILE_K, :])
            W_tiles_by_m.append(wt)
            bias_by_m.append(nl.load(bias[m_s:m_s + TILE_M]))

        for row_tile in nl.affine_range(n_row_tiles):
            oh_start = row_tile * ROW_TILE

            # ---- Load input strips for ALL k_tiles ONCE per row_tile ----
            # nl.par_dim(TILE_K) marks TILE_K=128 as the partition dimension.
            # First dim (n_tiles_c_in) is a batch dim indexed with Python int below.
            strips = nl.ndarray(
                shape=(n_tiles_c_in, nl.par_dim(TILE_K), STRIP_ROWS * input_width),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )
            for k_tile in nl.affine_range(n_tiles_c_in):
                k_s = k_tile * TILE_K
                for sr in nl.affine_range(STRIP_ROWS):
                    ih = oh_start + sr
                    strips[k_tile, :, sr * input_width:(sr + 1) * input_width] = nl.load(
                        X_flat[k_s:k_s + TILE_K, ih * input_width:(ih + 1) * input_width]
                    )

            # ---- Precompute X_tiles for ALL (k, fh, fw) combinations ----
            # Python range loops (compile-time) → k is a Python int, so strips[k, ...] works.
            # Each X_tile: (TILE_K, N_SPATIAL). Total: n_tiles_c_in*filter_h*filter_w tiles.
            # These are reused by ALL m_tiles, amortizing the SBUF copies across m_tiles.
            X_tiles_flat = []
            for k in range(n_tiles_c_in):
                for fh in range(filter_height):
                    for fw in range(filter_width):
                        X_tile = nl.ndarray(
                            shape=(TILE_K, N_SPATIAL),
                            dtype=X.dtype,
                            buffer=nl.sbuf,
                        )
                        for local_oh in nl.affine_range(ROW_TILE):
                            sc = (local_oh + fh) * input_width + fw
                            dc = local_oh * out_width
                            # strips[k, ...] uses Python int k → static slice selection
                            X_tile[:, dc:dc + out_width] = strips[k, :, sc:sc + out_width]
                        X_tiles_flat.append(X_tile)

            # ---- Compute output for each m_tile using preloaded X_tiles ----
            for m in range(n_tiles_c_out):
                m_s = m * TILE_M
                wt  = W_tiles_by_m[m]
                bt  = bias_by_m[m]

                # Single PSUM accumulates ALL (k, fh, fw) matmuls — one nl.copy at the end.
                psum = nl.zeros(
                    shape=(TILE_M, N_SPATIAL),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                tile_idx = 0
                for k in range(n_tiles_c_in):
                    for fh in range(filter_height):
                        for fw in range(filter_width):
                            filter_idx = fh * filter_width + fw
                            psum += nl.matmul(
                                wt[i_w.p, i_w.x, k, filter_idx],
                                X_tiles_flat[tile_idx][i_x.p, i_x.x],
                            )
                            tile_idx += 1

                result = nl.copy(psum, dtype=X.dtype)

                for local_oh in nl.affine_range(ROW_TILE):
                    src_col   = local_oh * out_width
                    global_oh = oh_start + local_oh

                    row_buf = nl.ndarray(
                        shape=(TILE_M, out_width),
                        dtype=X.dtype,
                        buffer=nl.sbuf,
                    )
                    row_buf[i_row.p, i_row.x] = result[i_row.p, src_col + i_row.x]
                    row_with_bias = nl.add(row_buf, bt)

                    if pool_size == 1:
                        nl.store(X_out[b, m_s:m_s + TILE_M, global_oh, :], value=row_with_bias)
                    else:
                        nl.store(conv_hbm[m_s:m_s + TILE_M, global_oh, :], value=row_with_bias)

        # ---- MaxPool phase (pool_size > 1 only) ----
        # Load full rows from conv_hbm to amortize HBM traffic across the spatial width.
        # For each output pool row ph:  load input rows h0=2*ph and h1=2*ph+1,
        # take element-wise max, then reduce adjacent column pairs.
        if pool_size > 1:
            i_pool_row = nl.mgrid[0:TILE_M, 0:out_width]
            i_pool_out = nl.mgrid[0:TILE_M, 0:out_pool_width]

            for m_tile in nl.affine_range(n_tiles_c_out):
                m_s = m_tile * TILE_M
                for ph in nl.affine_range(out_pool_height):
                    h0 = ph * 2
                    h1 = ph * 2 + 1

                    # Load two full conv rows: shape (TILE_M, out_width)
                    row0 = nl.load(conv_hbm[m_s:m_s + TILE_M, h0, :])
                    row1 = nl.load(conv_hbm[m_s:m_s + TILE_M, h1, :])

                    # Element-wise max across the two rows
                    row_max = nl.maximum(row0, row1)

                    # Reduce adjacent column pairs: out_pool_width = out_width // 2
                    even_cols = nl.ndarray(shape=(TILE_M, out_pool_width), dtype=X.dtype, buffer=nl.sbuf)
                    odd_cols  = nl.ndarray(shape=(TILE_M, out_pool_width), dtype=X.dtype, buffer=nl.sbuf)
                    for pw in nl.affine_range(out_pool_width):
                        even_cols[i_pool_out.p, pw] = row_max[i_pool_out.p, pw * 2]
                        odd_cols[i_pool_out.p, pw]  = row_max[i_pool_out.p, pw * 2 + 1]

                    pooled = nl.maximum(even_cols, odd_cols)
                    nl.store(X_out[b, m_s:m_s + TILE_M, ph, :], value=pooled)

    return X_out

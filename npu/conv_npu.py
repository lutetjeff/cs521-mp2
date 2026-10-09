import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A convolution kernel that you need to implement.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height
out_pool_width = out_width

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def conv2d(X, W, bias):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height
    out_pool_width = out_width
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0 and out_channels % 128 == 0
    assert filter_height == filter_width
    assert out_height > 0 and out_width > 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax
    c_out_pmax = nl.tile_size.gemm_stationary_fmax
    n_tiles_c_out = out_channels // c_out_pmax
    band_height = 32  # empirically chosen
    block_height = min(2, nl.tile_size.gemm_moving_fmax // out_width)

    # load all weights into SBUF.
    # for all the test harness's examples, weights easily fit into SBUF.
    # rather than im2col or streaming the weights from HBM, we load them only once into SBUF.
    # we load (128, cin / 128, kh, kw, cout) into SBUF. 
    w_sb = nl.ndarray(
        (c_in_pmax, n_tiles_c_in, filter_height, filter_width, out_channels),
        dtype=W.dtype,
        buffer=nl.sbuf,
    )
    # loop across sbuf-sized tiles of input and output channels
    for ci_group in nl.affine_range(n_tiles_c_in):
        for co_group in nl.affine_range(n_tiles_c_out):
            ci = ci_group * c_in_pmax
            co = co_group * c_out_pmax
            # load the tile into temporary buffer w_tile
            w_tile = nl.load(W[co:co + c_out_pmax, ci:ci + c_in_pmax, :, :])
            # transpose first two dimensions
            # nisa.nc_transpose is quite limiting but because we know kernels are 3x3
            # doing 9 transposes, one for each kernel location, is still fine
            # claude also didn't think there was a better way of doing this
            for kh in nl.affine_range(filter_height):
                for kw in nl.affine_range(filter_width):
                    w_sb[:, ci_group, kh, kw, co:co + c_out_pmax] = nl.copy(
                        nisa.nc_transpose(w_tile[:, :, kh, kw]), dtype=W.dtype
                    )
    # also load biases into SBUF. 
    bias_sb = nl.ndarray(
        (c_out_pmax, n_tiles_c_out, 1), dtype=bias.dtype, buffer=nl.sbuf
    )
    for co_group in nl.affine_range(n_tiles_c_out):
        co = co_group * c_out_pmax
        bias_sb[:, co_group, :] = nl.load(bias[co:co + c_out_pmax]).reshape((c_out_pmax, 1))

    # this is the first level of tiling, maximizing utilization of the SBUF
    # loop over output bands: we process groups of rows at a time with shape 
    # (128, cin / 128, band_height + filter_height - 1, width)
    for b in nl.affine_range(batch_size):
        for band_start in nl.static_range(0, out_height, band_height):
            band_rows = min(band_height, out_height - band_start)
            input_rows = band_rows + filter_height - 1

            # load the band into SBUF as (128, input_rows, width) tiles
            x_sb = nl.ndarray(
                (c_in_pmax, n_tiles_c_in, input_rows, input_width),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )
            for ci_group in nl.affine_range(n_tiles_c_in):
                ci = ci_group * c_in_pmax
                x_sb[:, ci_group, :, :] = nl.load(
                    X[b, ci:ci + c_in_pmax, band_start:band_start + input_rows, :]
                )

            # this is the second level of tiling, the blocks for the matrix units
            # on Trainium2, PSUM partition bank size is 2KiB, or 512 elements
            # so we size these blocks for accumulators of 512 elements
            # we use static_range to infer the range at JIT compile time
            for row in nl.static_range(0, band_rows, block_height):
                # edgecase handling for number of rows to process at a time
                block_rows = min(block_height, band_rows - row)
                # we still loop across the number of groups limited by the number of cout channels
                for co_group in nl.affine_range(n_tiles_c_out):
                    co = co_group * c_out_pmax
                    # initialize an accumulator of size (128, block_rows, out_width)
                    acc = nl.zeros(
                        (c_out_pmax, block_rows, out_width), dtype=nl.float32, buffer=nl.psum
                    )

                    # for each kernel position, perform a matrix multiplication with that kernel
                    # w_sb is (128 c_in, 128 c_out), we transpose it
                    # data is (128 c_in, block_rows, out_width)
                    for ci_group in nl.affine_range(n_tiles_c_in):
                        for kh in nl.affine_range(filter_height):
                            for kw in nl.affine_range(filter_width):
                                data = x_sb[:, ci_group, row + kh:row + kh + block_rows, kw:kw + out_width]
                                wts = w_sb[:, ci_group, kh, kw, co:co + c_out_pmax]
                                # nisa.nc_matmul computes w_sb.T @ data 
                                # it also flattens the free dimensions for data automatically
                                acc += nisa.nc_matmul(wts, data)

                    # add bias and write acc back to SBUF from PSUM
                    out_sb = nisa.tensor_scalar(acc, nl.add, bias_sb[:, co_group, :], dtype=X_out.dtype)
                    # store SBUF block back to HBM
                    nl.store(
                        X_out[b, co:co + c_out_pmax, band_start + row:band_start + row + block_rows, :],
                        value=out_sb,
                    )

    return X_out

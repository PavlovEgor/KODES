#ifndef KODES_BLOCK_REDUCE
#define KODES_BLOCK_REDUCE

#pragma once

#include <cuda_runtime.h>

#include "basic_types.cuh"

// Reductions over a whole block, for a solve where the block - not the thread -
// is the unit of integration.
//
// Every one of them reduces *and* broadcasts: the value comes back the same in
// every thread of the block. That is not a convenience, it is the invariant the
// collective solve rests on. A block-wide step size, extrapolation order and
// accept/reject decision are only safe while every thread reaches the same
// __syncthreads() the same number of times, and they only do that if the
// branches they take are computed from values all of them hold.
//
// For the same reason each of these is itself a collective call: the whole
// block has to reach it. A thread whose system is inactive - past the end of
// the batch, or too cold to react - still calls them, with a value that cannot
// win (see the `identity` arguments at each call site).

namespace kodes
{

// One entry per warp of the largest block the library will launch. Static
// rather than carved out of the dynamic shared memory, because the dynamic
// array belongs to pyJac's generated code and to the block's matrix, and a
// reduction is used from places that have neither in hand.
#define KODES_BLOCK_REDUCE_SLOTS (KODES_MAX_BLOCK_SIZE / KODES_WARP)

template<class T, class Op>
__device__ inline T blockReduce(T value, Op op)
{
    __shared__ T warpResult[KODES_BLOCK_REDUCE_SLOTS];

    const label lane = threadIdx.x & (KODES_WARP - 1);
    const label warp = threadIdx.x / KODES_WARP;
    const label warps = (blockDim.x + KODES_WARP - 1) / KODES_WARP;

    for (label offset = KODES_WARP/2; offset > 0; offset /= 2)
    {
        value = op(value, __shfl_down_sync(0xFFFFFFFFu, value, offset));
    }

    // The staging array is read back below, so the previous reduction using it
    // has to be finished before it is overwritten.
    __syncthreads();

    if (lane == 0)
    {
        warpResult[warp] = value;
    }

    __syncthreads();

    // Every thread folds the per warp results itself. Cheaper than reducing
    // once and broadcasting, since warps is at most 32 and the array is
    // already in shared memory, and it leaves the result in a register of
    // every thread rather than in one it would have to be read out of.
    T result = warpResult[0];

    for (label i = 1; i < warps; ++i)
    {
        result = op(result, warpResult[i]);
    }

    return result;
}

// The largest value any thread of the block passed in
__device__ inline scalar blockMax(const scalar value)
{
    return blockReduce
    (
        value,
        [] (const scalar a, const scalar b) { return a > b ? a : b; }
    );
}

// The smallest value any thread of the block passed in
__device__ inline scalar blockMin(const scalar value)
{
    return blockReduce
    (
        value,
        [] (const scalar a, const scalar b) { return a < b ? a : b; }
    );
}

// Whether any thread of the block passed true
__device__ inline bool blockAny(const bool flag)
{
    return blockReduce
    (
        label(flag ? 1 : 0),
        [] (const label a, const label b) { return a > b ? a : b; }
    ) != 0;
}

// The highest threadIdx.x among the threads that passed true, or -1 if none
// did. Which thread that is matters: the balanced order runs from the least to
// the most stiff system, so the last active lane of a block holds its stiffest
// one - see kJacobianAtStiffestLane in Seulex.cu.
__device__ inline label blockMaxActiveLane(const bool active)
{
    return blockReduce
    (
        active ? label(threadIdx.x) : label(-1),
        [] (const label a, const label b) { return a > b ? a : b; }
    );
}

// The largest value any thread of the block passed in, together with the index
// that came with it; ties go to the smallest index. Both are returned in every
// thread. Written out rather than built from blockMax and blockReduce, because
// a pivot search runs once per column of the factorization and the pair costs
// one pass of barriers instead of two.
__device__ inline void blockArgMax(scalar& value, label& index)
{
    __shared__ scalar warpValue[KODES_BLOCK_REDUCE_SLOTS];
    __shared__ label  warpIndex[KODES_BLOCK_REDUCE_SLOTS];

    const label lane = threadIdx.x & (KODES_WARP - 1);
    const label warp = threadIdx.x / KODES_WARP;
    const label warps = (blockDim.x + KODES_WARP - 1) / KODES_WARP;

    for (label offset = KODES_WARP/2; offset > 0; offset /= 2)
    {
        const scalar otherValue = __shfl_down_sync(0xFFFFFFFFu, value, offset);
        const label  otherIndex = __shfl_down_sync(0xFFFFFFFFu, index, offset);

        if (otherValue > value || (otherValue == value && otherIndex < index))
        {
            value = otherValue;
            index = otherIndex;
        }
    }

    __syncthreads();

    if (lane == 0)
    {
        warpValue[warp] = value;
        warpIndex[warp] = index;
    }

    __syncthreads();

    value = warpValue[0];
    index = warpIndex[0];

    for (label i = 1; i < warps; ++i)
    {
        if (warpValue[i] > value || (warpValue[i] == value && warpIndex[i] < index))
        {
            value = warpValue[i];
            index = warpIndex[i];
        }
    }
}

}

#endif

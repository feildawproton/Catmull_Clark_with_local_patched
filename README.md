# Catmull_Clark_with_local_patches
Another old school project.  This one tests how memory optimization could affect Catmull-Clark algorithm

# Catmull-Clark with CUDA and openMP

## Sub-division surfaces
- Splines
  - “Infinite” resolution
  - Smooth curves
  - Hard to animate and texture
- Meshes
  - Easy to model and texture
  - Easy to animate
  - Rough/Imperfect surface
  - Large footprint
- Subdivision surface
  - Combine both
  - Mesh acts as B-spline
![image](https://user-images.githubusercontent.com/56926839/162259000-2f1ad240-e5c0-43dc-91ad-596f4b537cc1.png)

## Real-time subsurf motivation
- Currently used in film and modeling (actually this has probably changed since I wrote this all the way back in 2017)
  - Pixar popularized
  - Saves disk space
    - part of modifier stack
  - SubSurf is applied by CPU
  - Result is in RAM
- GPU usage
  - There is current interest in applying subsurf on GPU
  - Saves RAM and bandwidth
  - Don’t need the result back
    - Push out to screem
##  Catmull-Clark
- Meshes are:
  - Vertices
  - Faces
  - Contain indices to vertex array
  - Edges
  - Contain indices to face array
  - Arbitrary meshes are not aligned in memory in accordance to their spatial relationship
- Algorithm
  - Calculate face average points F
  - Calculate edge average points R
    - Uses results from F (sync locally here)
  - Calculate barycenter of vertices P
    - Uses results from F
  - Close faces
  - Repeat as desired

## Parallelization Approach
- Some calculations require data from a neighbor
- We can read that data and perform a duplicate calculation
- Break mesh into patches
  - Let individual threads submit work to the GPU
- The patches will be divided into the thread blocks for the GPU kernels that do the calculations  
- Theory: GPU Multiprocessors spend a lot of time waiting on reads; however they also have a good scheduler
  - Keep GPUs busy and task switching with multiple threads

## Optimization approach
- Wondered if we could lay mesh data in regular grids
  - Theory: more cash friendly if neighbors are near each other
- This is what I came up with:
  - Assume quads only – a typical requirement
  - Assume manifold mesh – no wormholes
  - Break mesh into patches of regular matrices
    - Using non-degree 4 vertices as a guide

![image](https://user-images.githubusercontent.com/56926839/162259870-02021bf2-440b-4710-a152-831c1d88831d.png)

## Implementation
- CUDA
  - NVCC
- OpenMP
  - Can combine with CUDA through “streams”
  - NCVV supports openMP with
    -Xcompiler -fopenmp
    --default-dtream per-thread
- Batch Size Macro – 4 compiled version
  - (4x4, 8x8, 16x16, 32x32
- openMP threads and mesh size are command line options

## Compute Platform
- Used my of computer
- OS: Linux Mint 18.1 Cinnamon 64-bit
- CPU: Intel i7-6700k
  - 4 physical core, 8 logical core @4 GHz
- 16 GB DDR4
- GPU: Nvidia GTX 1080
  - Compute Capability 6.1
  - 2560 CUDA Cores @ 2GHz
  - 8 GB GDDR5 @ “10GHz”

## Experiments
- Each Ran multiple times
- Six mesh sizes
  - 11664(108x108), 104876(324*324), 101604(1008x1008), 10036224(3168x3168), 100160064(10008x10008), 1000583424(31632x31632)
- Four Paths Counts/Thread counts
  - 1,4,9,16
  - Exceeds core count (8) – job is just to submit work to GPU
  - Divides up the mesh
- Four Block Sizes
  - 16(4x4), 64(8x8), 256(16*16), 1024(32x32)
  - Chosen to divide well by the SM
  - Divide up the patches
- Used high resolution time available with C11

## Patch/Thread Resultsd 32x32
![image](https://user-images.githubusercontent.com/56926839/162260420-9a47d4c4-2551-492d-9a22-1273a96a27c8.png)

## Block Size and Block Threads Results
- No noticeable pattern
- Each calculation is highly independent
  - Perhaps we were unable to capitalize on the shared memory afforded by larger block sizes

## Optimization Results
- Internally patches had random neighbors
- Did not duplicate the problem of random neighbors between patches
  - The effect we see here is just from the GPU
![image](https://user-images.githubusercontent.com/56926839/162260634-a219b1e4-3a03-4718-a0be-0fe769a27b2a.png)

## Analysis
- Results scaled with thread count on the largest mesh that didn’t fail
  - The CPU threads were thus keeping the GPU Streaming Multiprocessors busy
- Results did not scale with block size
  - This is perhaps because the algorithm couldn’t benefit from the shared memory
- Results of optimization were promising
  - The tested irregular mesh was just irregular with the patch
  - Perhaps results would be more striking otherwise

## Future (perhaps not...)
- I’d like to combine this with an actual graphics application
- Combine with tesselation
- Explore use with progressive collision detection

















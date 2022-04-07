/*
example compile $nvcc -Xcompiler -fopenmp --defaul-stream per-thread --ptxas-options=-v final_regularQuadMesh.cu -o meh.out
the usage with multiple threads comes from the NVidia developers blog
DON'T FORGET GET TO USE sudo WHEN YOU RUN THE PROGRAM 
*/

#include <stdio.h>
#include "omp.h" 
#include <time.h>
#include <stdlib.h>

//macros because grossness and to get help from compiler with --ptxas-options=-v
#define BLOCK_VERT_DIM 4

#define BLOCK_FACE_DIM (BLOCK_VERT_DIM - 1)

#define BLOCK_EDGE_HRZNTL_HEIGHT BLOCK_VERT_DIM
#define BLOCK_EDGE_HRZNTL_WIDTH BLOCK_FACE_DIM

#define BLOCK_EDGE_VRTCL_HEIGHT BLOCK_FACE_DIM
#define BLOCK_EDGE_VRTCL_WIDTH BLOCK_VERT_DIM

#define BASE_MESH_DIM 840

//define SHARING

typedef struct Vert
{
    float w; //in case we want it
    float x;
    float y;
    float z;
} Vert;

/*
for meshes that cannot be broken down into regular matrices OR it's not work it
access as rVerts.v_p[width*row + col]
size is height * width
*/
typedef struct reg_Verts
{
    int height; //rows
    int width; //cols
    Vert* v_p;
} reg_Verts;

//vert = *(rVerts + i) == rVerts[i]
typedef struct quad_Face
{  
    int v1_i;
    int v2_i;
    int v3_i;
    int v4_i;
} quad_Face;

typedef struct quad_Faces
{
    int height;
    int width;
    quad_Face* f_p;
} quad_Faces;

typedef struct mani_Edge
{
    int f1_i;
    int f2_i;
} mani_Edge;

typedef struct mani_Edges
{
    int height;
    int width;
    mani_Edge* e_p;
} mani_Edges;

//here we break the mesh into regular patches
typedef struct Patch
{
    reg_Verts verts;
    quad_Faces faces;
    mani_Edges edges_hrzntl;
    mani_Edges edges_vrtcl;
} Patch;

typedef struct Mesh
{
    int p_count;
    Patch* p_p; 
} Mesh;

__global__ void gpu_face_points(quad_Faces faces, reg_Verts verts)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row > faces.height || col > faces.width) return;

    quad_Face myFace = faces.f_p[row * faces.width + col];

    Vert v1 = verts.v_p[myFace.v1_i];
    Vert v2 = verts.v_p[myFace.v2_i];
    Vert v3 = verts.v_p[myFace.v3_i];
    Vert v4 = verts.v_p[myFace.v4_i];

    Vert result;
    result.w = (v1.w + v2.w + v3.w + v4.w)/4;
    result.x = (v1.x + v2.x + v3.x + v4.x)/4;
    result.y = (v1.y + v2.y + v3.y + v4.y)/4;
    result.z = (v1.z + v2.z + v3.z + v4.z)/4;

    //just throw away result for now
    //face_verts.v_p[row * faces.width + col] = result;
}

__global__ void gpu_edge_points(mani_Edges edges, quad_Faces faces, reg_Verts verts)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row > edges.height || col > edges.width) return;

    mani_Edge myEdge = edges.e_p[row*edges.width + col];

    quad_Face f1 = faces.f_p[myEdge.f1_i];
    quad_Face f2 = faces.f_p[myEdge.f2_i];

    Vert v1 = verts.v_p[f1.v1_i];
    Vert v2 = verts.v_p[f1.v2_i];
    Vert v3 = verts.v_p[f1.v3_i];
    Vert v4 = verts.v_p[f1.v4_i];
    
    Vert f1_center;
    f1_center.w = (v1.w + v2.w + v3.w + v4.w)/4;
    f1_center.x = (v1.x + v2.x + v3.x + v4.x)/4;
    f1_center.y = (v1.y + v2.y + v3.y + v4.y)/4;
    f1_center.z = (v1.z + v2.z + v3.z + v4.z)/4;

    v1 = verts.v_p[f1.v1_i];
    v2 = verts.v_p[f1.v2_i];
    v3 = verts.v_p[f1.v3_i];
    v4 = verts.v_p[f1.v4_i];
    
    Vert f2_center;
    f2_center.w = (v1.w + v2.w + v3.w + v4.w)/4;
    f2_center.x = (v1.x + v2.x + v3.x + v4.x)/4;
    f2_center.y = (v1.y + v2.y + v3.y + v4.y)/4;
    f2_center.z = (v1.z + v2.z + v3.z + v4.z)/4;
    
    Vert edge_result;
    edge_result.w = (f1_center.w + f2_center.w) / 2;
    edge_result.x = (f1_center.w + f2_center.x) / 2;
    edge_result.y = (f1_center.w + f2_center.y) / 2;
    edge_result.z = (f1_center.w + f2_center.z) / 2;

    //just through result for now
    //edgeVerts.v_p[row*edgeVerts.width + col] = result;
}

__global__ void gpu_vert_smooth(reg_Verts vert)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    Vert P = vert.v_p[row * vert.width + col];

    //need t pass these in or read
    Vert F1;
    Vert F2;
    Vert F3;
    Vert F4;
    
    F1.w = F2.w = F3.w = F4.w = 0;
    F1.x = F2.x = F3.x = F4.x = (float)col;
    F1.y = F2.y = F3.y = F4.y = (float)row;
    F1.z = F2.z = F3.z = F4.z = (float)row*col;

    Vert R1;
    Vert R2;
    Vert R3;
    Vert R4;
    
    R1.w = R2.w = R3.w = R4.w = 0;
    R1.x = R2.x = R3.x = R4.x = (float)row;
    R1.y = R2.y = R3.y = R4.y = (float)col;
    R1.z = R2.z = R3.z = R4.z = (float)row*col;

    Vert F;
    F.w = (F1.w + F2.w + F3.w + F4.w) / 4;
    F.x = (F1.x + F2.x + F3.x + F4.x) / 4;
    F.y = (F1.y + F2.y + F3.y + F4.y) / 4;
    F.z = (F1.z + F2.z + F3.z + F4.z) / 4;

    Vert R;
    R.w = (R1.w + R2.w + R3.w + R4.w) / 4;
    R.x = (R1.x + R2.x + R3.x + R4.x) / 4;
    R.y = (R1.y + R2.y + R3.y + R4.y) / 4;
    R.z = (R1.z + R2.z + R3.z + R4.z) / 4;

    int n = 9;

    Vert result;

    result.w = (F.w + 2*R.w + (n - 3)*(P.w)) / 4; //4 face points assumed for now
    result.x = (F.w + 2*R.x + (n - 3)*(P.x)) / 4; //4 face points assumed for now
    result.y = (F.w + 2*R.y + (n - 3)*(P.y)) / 4; //4 face points assumed for now
    result.z = (F.z + 2*R.z + (n - 3)*(P.z)) / 4; //4 face points assumed for now

    
    //give the result out
}

void host_subDiv_patch(const Patch pat, unsigned int patchID, unsigned int cpuThreads)
{
    //create a copy on the device
    Patch gpu_cpy;

    //VERTS cpy
    gpu_cpy.verts.height = pat.verts.height;
    gpu_cpy.verts.width = pat.verts.width;
    size_t vert_size = pat.verts.height * pat.verts.width * sizeof(Vert);
    cudaError_t err_cpy = cudaMalloc(&gpu_cpy.verts.v_p, vert_size);
    if(strcmp(cudaGetErrorString(err_cpy), "no error") != 0)
        printf("patch %d cuda malloc gpu_cpy verts with %s\n", patchID, cudaGetErrorString(err_cpy));
    cudaMemcpy(gpu_cpy.verts.v_p, pat.verts.v_p, vert_size, cudaMemcpyHostToDevice);

    //FACES cpy
    gpu_cpy.faces.height = pat.faces.height;
    gpu_cpy.faces.width = pat.faces.width;
    size_t face_size = pat.faces.height * pat.faces.width * sizeof(quad_Face);
    err_cpy = cudaMalloc(&gpu_cpy.faces.f_p, face_size);
    if(strcmp(cudaGetErrorString(err_cpy), "no error") != 0)
        printf("patch %d cuda malloc gpu_cpy faces with %s\n", patchID, cudaGetErrorString(err_cpy));
    cudaMemcpy(gpu_cpy.faces.f_p, pat.faces.f_p, face_size, cudaMemcpyHostToDevice);

    //EDGES HORIZONTAL cpy
    gpu_cpy.edges_hrzntl.height = pat.edges_hrzntl.height;
    gpu_cpy.edges_hrzntl.width = pat.edges_hrzntl.width;
    size_t edge_hrzntl_size = pat.edges_hrzntl.height * pat.edges_hrzntl.width * sizeof(mani_Edge);
    err_cpy = cudaMalloc(&gpu_cpy.edges_hrzntl.e_p, edge_hrzntl_size);
    if(strcmp(cudaGetErrorString(err_cpy), "no error") != 0)
        printf("patch %d cuda malloc gpu_cpy horizontal edges with %s\n", patchID, cudaGetErrorString(err_cpy));
    cudaMemcpy(gpu_cpy.edges_hrzntl.e_p, pat.edges_hrzntl.e_p, edge_hrzntl_size, cudaMemcpyHostToDevice);

    //EDGES vertical cpy
    gpu_cpy.edges_vrtcl.height = pat.edges_vrtcl.height;
    gpu_cpy.edges_vrtcl.width = pat.edges_vrtcl.width;
    size_t edge_vrtcl_size = pat.edges_vrtcl.height * pat.edges_vrtcl.width * sizeof(mani_Edge);
    err_cpy = cudaMalloc(&gpu_cpy.edges_vrtcl.e_p, edge_vrtcl_size);
    if(strcmp(cudaGetErrorString(err_cpy), "no error") != 0)
        printf("patch %d cuda malloc gpu_cpy vertical EDGES with %s\n", patchID, cudaGetErrorString(err_cpy));
    cudaMemcpy(gpu_cpy.edges_vrtcl.e_p, pat.edges_vrtcl.e_p, edge_vrtcl_size, cudaMemcpyHostToDevice);


    dim3 dimBlock(BLOCK_FACE_DIM, BLOCK_FACE_DIM);
    dim3 dimGrid(gpu_cpy.faces.width / BLOCK_FACE_DIM, gpu_cpy.faces.height / BLOCK_FACE_DIM);    
    gpu_face_points<<<dimGrid, dimBlock>>>(gpu_cpy.faces, gpu_cpy.verts);


    dim3 dimBlockE_h(BLOCK_EDGE_HRZNTL_HEIGHT, BLOCK_EDGE_HRZNTL_WIDTH);
    dim3 dimGridE_h(gpu_cpy.edges_hrzntl.width / BLOCK_EDGE_HRZNTL_WIDTH, gpu_cpy.edges_hrzntl.height / BLOCK_EDGE_HRZNTL_HEIGHT); 
    gpu_edge_points<<<dimGridE_h, dimBlockE_h>>>(gpu_cpy.edges_hrzntl, gpu_cpy.faces, gpu_cpy.verts);      

    dim3 dimBlockE_v(BLOCK_VERT_DIM, BLOCK_VERT_DIM);
    dim3 dimGridE_v(gpu_cpy.verts.width / BLOCK_VERT_DIM, gpu_cpy.verts.height / BLOCK_VERT_DIM); 
    gpu_vert_smooth<<<dimGridE_v, dimBlockE_v>>>(gpu_cpy.verts);

    cudaStreamSynchronize(0);



    //copy back when done if you want

    cudaFree(gpu_cpy.verts.v_p);
    cudaFree(gpu_cpy.faces.f_p);
    cudaFree(gpu_cpy.edges_hrzntl.e_p);
    cudaFree(gpu_cpy.edges_vrtcl.e_p);
    //cudaFree(face_centers.v_p);
}  

//i found this on github thanks to diablonea and his/her example timespec_diff.c
void diff_times(struct timespec* start, struct timespec* stop, struct timespec *result)
{
    //changed the name and formating of input but the following is a SHAMEFUL copy paste...
    if ((stop->tv_nsec - start->tv_nsec) < 0) {
        result->tv_sec = stop->tv_sec - start->tv_sec - 1;
        result->tv_nsec = stop->tv_nsec - start->tv_nsec + 1000000000;
    } else {
        result->tv_sec = stop->tv_sec - start->tv_sec;
        result->tv_nsec = stop->tv_nsec - start->tv_nsec;
    }
    return;
}

Patch create_patch(unsigned int verts_dim, unsigned int t)
{
    int faces_dim = verts_dim - 1;

    Patch pat;

    pat.verts.height = verts_dim;
    pat.verts.width = verts_dim;
    pat.verts.v_p = (Vert *)malloc(verts_dim * verts_dim * sizeof(Vert));
        
    pat.faces.height = faces_dim;
    pat.faces.width = faces_dim;
    pat.faces.f_p = (quad_Face *)malloc(faces_dim * faces_dim * sizeof(quad_Face));

    pat.edges_hrzntl.height = verts_dim;
    pat.edges_hrzntl.width = faces_dim;
    pat.edges_hrzntl.e_p = (mani_Edge *)malloc(verts_dim * faces_dim * sizeof(mani_Edge));

    pat.edges_vrtcl.height = faces_dim;
    pat.edges_vrtcl.width = verts_dim;
    pat.edges_vrtcl.e_p = (mani_Edge *)malloc(faces_dim * verts_dim * sizeof(mani_Edge));

/*
    printf("pat verts %d %d\n", pat.verts.height, pat.verts.width);
    printf("pat ed hr %d %d\n", pat.edges_hrzntl.height, pat.edges_hrzntl.width);
    printf("pat ed v %d %d\n", pat.edges_vrtcl.height, pat.edges_vrtcl.width);
    printf("pat faces %d %d\n", pat.faces.height, pat.faces.width);
*/

    for(unsigned int m = 0; m < pat.verts.height; m++)
    {
        for(unsigned int n = 0; n < pat.verts.width; n++)
        {
            Vert vert_Value;
            vert_Value.w = 0;
            vert_Value.x = n/1000;
            vert_Value.y = m/1000;
            vert_Value.z = (m * n) / 1000;

            pat.verts.v_p[m * pat.verts.width + n] = vert_Value;
        }
    }

    //there are fewer faces and finding the right vertices requires a little bit if thinking
    for(unsigned int m = 0; m < pat.faces.height; m++)
    {
        for(unsigned int n = 0; n < pat.faces.width; n++)
        {
            quad_Face face_value;            

            face_value.v1_i = (m*n) - m * pat.verts.width + n;
            face_value.v2_i = (m*n) - face_value.v3_i + 1;
            face_value.v3_i = (m*n) - face_value.v1_i + pat.verts.width;
            face_value.v4_i = (m*n) - face_value.v3_i + 1;
                        
            pat.faces.f_p[m * pat.faces.width + n] = face_value;
        }
    }

    //more kinda funky stuff
    for(int i = 0; i < pat.edges_hrzntl.height*pat.edges_hrzntl.width; i++)
    {
        mani_Edge edge_value;
        edge_value.f1_i = i;
        int j = (pat.edges_hrzntl.height*pat.edges_hrzntl.width) - i - pat.edges_hrzntl.width;

        if(j < 0) j += (pat.edges_hrzntl.height * pat.edges_hrzntl.width);

        edge_value.f2_i = (pat.edges_hrzntl.height*pat.edges_hrzntl.width) - j;

        *(pat.edges_hrzntl.e_p + i) = edge_value;
    }    

    for(int m = 0; m < pat.edges_vrtcl.height; m++)
    {
        for(int n = 0; n < pat.edges_vrtcl.width; n++)
        {
            mani_Edge edge_value;
            //all the way on the right
            if(n == pat.faces.width)
            {
                edge_value.f1_i = (m*n) - m*pat.faces.width;
                edge_value.f2_i = (m*n) - m*pat.faces.width + (n - 1);
            }
            else
            {
                edge_value.f1_i = (m*n) - m*pat.faces.width + n;
                if(n == 0)
                    edge_value.f2_i = (m*n) - pat.faces.width*(m + 1) - 1;
                else
                    edge_value.f2_i = (m*n) - m*pat.faces.width + (n-1);
                
            }
            
            pat.edges_vrtcl.e_p[m * pat.edges_vrtcl.width + n] = edge_value;
        }   
    }

    return pat;                         
}

int main(int argc, char** argv)
{
    int mesh_verts = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    
    int mesh_dims = sqrt(mesh_verts);

    int patch_verts = mesh_verts / num_threads;
    int patch_dim = sqrt(patch_verts);
    
    omp_set_num_threads(num_threads);

    printf("mesh vert size id %d with dimension %d and patch size is %d with dimension %d with %d patch threads\n", mesh_verts, mesh_dims, patch_verts, patch_dim, num_threads);
   
    int check = (patch_dim*patch_dim)*num_threads;
    printf("this comes out to a total checked mesh size of %d\n", check);

    //timing
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    char buff[100];
    //strftime(buff, sizeof buff, "%D %T", gmtime(&ts.tv_sec));
    //printf("WALL CLOCK BEFORE CPU ALLOC: %s.%09ld UTC\n", buff, ts.tv_nsec);

    unsigned int t;
    srand(t);

    //parallel allocation
    int num_thread_patches;
    Patch* pats;
    #pragma omp parallel shared(num_thread_patches, pats, t)
    {
        int thread_patchID = omp_get_thread_num();
        if(thread_patchID == 0)
        {
            num_thread_patches = omp_get_num_threads();
            printf("number of threads = %d\n", num_thread_patches);
            pats = (Patch*)malloc(num_thread_patches * sizeof(Patch));
            printf("with the number of patches being = %d\n", num_thread_patches);
        }
        
        Patch pat = create_patch(patch_dim, t);
        pats[thread_patchID] = pat;
    }

    
    printf("created with %d patches\n", num_thread_patches);

    struct timespec tCPUalloc;
    timespec_get(&tCPUalloc, TIME_UTC);
    struct timespec tCPUalloc_duration;
    diff_times(&ts, &tCPUalloc,  &tCPUalloc_duration); 
    strftime(buff, sizeof buff, "%D %T", gmtime(&tCPUalloc_duration.tv_sec));
    printf("CPU ALLOC DURATION: %s.%09ld UTC\n", buff, tCPUalloc_duration.tv_nsec);


    //do the stuff on the gpu
    #pragma omp parallel shared(pats)
    {   
        int thread_patchID = omp_get_thread_num();
        Patch pat = pats[thread_patchID];
        host_subDiv_patch(pat, thread_patchID, num_thread_patches);
    }
    

    struct timespec tGPUcompute;
    timespec_get(&tGPUcompute, TIME_UTC);
    struct timespec tGPUcompute_duration;

    diff_times(&tCPUalloc, &tGPUcompute,  &tGPUcompute_duration);
    
    strftime(buff, sizeof buff, "%D %T", gmtime(&tGPUcompute_duration.tv_sec));
    printf("DURATION OF GPU COPY, EXECUTE, AND RETURN: %s.%09ld UTC\n", buff, tGPUcompute_duration.tv_nsec);    

    //free
    for(unsigned int n = 0; n < num_thread_patches; n++)
    {
        free(pats[n].edges_vrtcl.e_p);
        free(pats[n].edges_hrzntl.e_p);
        free(pats[n].faces.f_p);
        free(pats[n].verts.v_p);  
    } 

    //free(pat.edges_vrtcl.e_p);
    //free(pat.edges_hrzntl.e_p);
    //free(pat.faces.f_p);
    //free(pat.verts.v_p);

    return 0;
}











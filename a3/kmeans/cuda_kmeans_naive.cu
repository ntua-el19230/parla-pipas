#include <stdio.h>
#include <stdlib.h>

#include "kmeans.h"
#include "alloc.h"
#include "error.h"

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        // cudaGetErrorString() isn't always very helpful. Look up the error
        // number in the cudaError enum in driver_types.h in the CUDA includes
        // directory for a better explanation.
        error("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif

__device__ int get_tid(){
    /* TODO: Calculate 1-Dim global ID of a thread */
	return blockDim.x * blockIdx.x + threadIdx.x;
}

/* square of Euclid distance between two multi-dimensional points */
__host__ __device__ inline static
float euclid_dist_2(int    numCoords,
                    int    numObjs,
                    int    numClusters,
                    float *objects,     // [numObjs][numCoords]
                    float *clusters,    // [numClusters][numCoords]
                    int    objectId,
                    int    clusterId)
{
    int i;
    float ans=0.0;

	/* TODO: Calculate the euclid_dist of elem=objectId of objects from elem=clusterId from clusters*/
    for (i=0; i<numCoords; i++)
        ans += (objects[objectId*numCoords + i] - clusters[clusterId*numCoords + i]) *
               (objects[objectId*numCoords + i] - clusters[clusterId*numCoords + i]);

    return(ans);
}

__global__ static
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          float *objects,           //  [numObjs][numCoords]
                          float *deviceClusters,    //  [numClusters][numCoords]
                          int *deviceMembership,          //  [numObjs]
                          float *devdelta)
{

	/* Get the global ID of the thread. */
    int tid = get_tid();

	/* TODO: Maybe something is missing here... should all threads run this? */
    if (tid < numObjs) {
        int   index, i;
        float dist, min_dist;

        /* find the cluster id that has min distance to object */
        index = 0;
        /* TODO: call min_dist = euclid_dist_2(...) with correct objectId/clusterId */
        min_dist = euclid_dist_2(numCoords, numObjs, numClusters, objects, deviceClusters, tid, 0);

        for (i=1; i<numClusters; i++) {
            /* TODO: call dist = euclid_dist_2(...) with correct objectId/clusterId */
            dist = euclid_dist_2(numCoords, numObjs, numClusters, objects, deviceClusters, tid, i);

            /* no need square root */
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index    = i;
            }
        }

        if (deviceMembership[tid] != index) {
        	/* TODO: Maybe something is missing here... is this write safe? */
            // (*devdelta)+= 1.0;
            atomicAdd(devdelta, 1.0);
        }

        /* assign the deviceMembership to object objectId */
        deviceMembership[tid] = index;
    }
}

//
//  ----------------------------------------
//  DATA LAYOUT
//
//  objects         [numObjs][numCoords]
//  clusters        [numClusters][numCoords]
//  newClusters     [numClusters][numCoords]
//  deviceObjects   [numObjs][numCoords]
//  deviceClusters  [numClusters][numCoords]
//  ----------------------------------------
//
/* return an array of cluster centers of size [numClusters][numCoords]       */
void kmeans_gpu(	float *objects,      /* in: [numObjs][numCoords] */
		               	int     numCoords,    /* no. features */
		               	int     numObjs,      /* no. objects */
		               	int     numClusters,  /* no. clusters */
		               	float   threshold,    /* % objects change membership */
		               	long    loop_threshold,   /* maximum number of iterations */
		               	int    *membership,   /* out: [numObjs] */
						float * clusters,   /* out: [numClusters][numCoords] */
						int blockSize)
{
    double timing = wtime(), timing_internal, timer_min = 1e42, timer_max = 0;
	int    loop_iterations = 0;
    int      i, j, index, loop=0;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    float  delta = 0, *dev_delta_ptr;          /* % of objects change their clusters */
    float  **newClusters = (float**) calloc_2d(numClusters, numCoords, sizeof(float));

    float *deviceObjects;
    float *deviceClusters;
    int *deviceMembership;

    printf("\n|-------------Naive GPU Kmeans--------------|\n\n");


    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    timing = wtime() - timing;
    printf("t_alloc: %lf ms\n\n", 1000*timing);
    timing = wtime();

    const unsigned int numThreadsPerClusterBlock = (numObjs > blockSize)? blockSize: numObjs;
    /* TODO: Calculate Grid size, e.g. number of blocks. */
    const unsigned int numClusterBlocks = (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;
    const unsigned int clusterBlockSharedDataSize = 0;

    checkCuda(cudaMalloc(&deviceObjects, numObjs*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&deviceClusters, numClusters*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
    checkCuda(cudaMalloc(&dev_delta_ptr, sizeof(float)));

    timing = wtime() - timing;
    printf("t_alloc_gpu: %lf ms\n\n", 1000*timing);
    timing = wtime();

    checkCuda(cudaMemcpy(deviceObjects, objects,
              numObjs*numCoords*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceMembership, membership,
              numObjs*sizeof(int), cudaMemcpyHostToDevice));
    timing = wtime() - timing;
    printf("t_get_gpu: %lf ms\n\n", 1000*timing);
    timing = wtime();

    do {
        timing_internal = wtime();

		/* GPU part: calculate new memberships */

        /* TODO: Copy clusters to deviceClusters
        checkCuda(cudaMemcpy(...)); */
        checkCuda(cudaMemcpy(deviceClusters, clusters,
                  numClusters*numCoords*sizeof(float), cudaMemcpyHostToDevice));

        checkCuda(cudaMemset(dev_delta_ptr, 0, sizeof(float)));

		//printf("Launching find_nearest_cluster Kernel with grid_size = %d, block_size = %d, shared_mem = %d KB\n", numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize/1000);
        find_nearest_cluster
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numCoords, numObjs, numClusters,
             deviceObjects, deviceClusters, deviceMembership, dev_delta_ptr);

        cudaDeviceSynchronize(); checkLastCudaError();
		//printf("Kernels complete for itter %d, updating data in CPU\n", loop);

		/* TODO: Copy deviceMembership to membership
        checkCuda(cudaMemcpy(...)); */
        checkCuda(cudaMemcpy(membership, deviceMembership,
                  numObjs*sizeof(int), cudaMemcpyDeviceToHost));

    	/* TODO: Copy dev_delta_ptr to &delta
        checkCuda(cudaMemcpy(...)); */
        checkCuda(cudaMemcpy(&delta, dev_delta_ptr,
                  sizeof(float), cudaMemcpyDeviceToHost));

		/* CPU part: Update cluster centers*/

        for (i=0; i<numObjs; i++) {
            /* find the array index of nestest cluster center */
            index = membership[i];

            /* update new cluster centers : sum of objects located within */
            newClusterSize[index]++;
            for (j=0; j<numCoords; j++)
                newClusters[index][j] += objects[i*numCoords + j];
        }

        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 0)
                    clusters[i*numCoords + j] = newClusters[i][j] / newClusterSize[i];
                newClusters[i][j] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }

        delta /= numObjs;
       	//printf("delta is %f - ", delta);
        loop++;
        //printf("completed loop %d\n", loop);
		timing_internal = wtime() - timing_internal;
		if ( timing_internal < timer_min) timer_min = timing_internal;
		if ( timing_internal > timer_max) timer_max = timing_internal;
    } while (delta > threshold && loop < loop_threshold);

    timing = wtime() - timing;
    printf("nloops = %d  : total = %lf ms\n\t-> t_loop_avg = %lf ms\n\t-> t_loop_min = %lf ms\n\t-> t_loop_max = %lf ms\n\n|-------------------------------------------|\n",
    	loop, 1000*timing, 1000*timing/loop, 1000*timer_min, 1000*timer_max);

	char outfile_name[1024] = {0};
	sprintf(outfile_name, "Execution_logs/Sz-%ld_Coo-%d_Cl-%d.csv", numObjs*numCoords*sizeof(float)/(1024*1024), numCoords, numClusters);
	FILE* fp = fopen(outfile_name, "a+");
	if(!fp) error("Filename %s did not open succesfully, no logging performed\n", outfile_name);
	fprintf(fp, "%s,%d,%lf,%lf,%lf\n", "Naive", blockSize, timing/loop, timer_min, timer_max);
	fclose(fp);
    checkCuda(cudaFree(deviceObjects));
    checkCuda(cudaFree(deviceClusters));
    checkCuda(cudaFree(deviceMembership));

    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return;
}


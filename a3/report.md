<div align='center'>
  <img src='../a2/ntua.svg' width=200>
  <br/>
  <h3>ΕΘΝΙΚΟ ΜΕΤΣΟΒΙΟ ΠΟΛΥΤΕΧΝΙΟ</h3>
  <h4>ΣΧΟΛΗ ΗΛΕΚΤΡΟΛΟΓΩΝ ΜΗΧΑΝΙΚΩΝ ΚΑΙ ΜΗΧΑΝΙΚΩΝ ΥΠΟΛΟΓΙΣΤΩΝ</h4>
  <h5>Συστήματα Παράλληλης Επεξεργασίας</h5>
  <h6>Άσκηση 3: Παραλληλοποίηση και βελτιστοποίηση αλγορίθμων σε επεξεργαστές γραφικών</h6>
</div>

---

| Όνομα | Επώνυμο | Α.Μ. |
|-------|---------|------|
| Αλτάν    | Αβτζή   | 03119241 |
| Τζόναταν | Λουκάι  | 03119230 |
| Σταύρος  | Λαζάρου | 03112642 |

<br/>
<br/>

## Αλγόριθμος K-means
### Υλοποίηση Naive version
1.Υλοποίηση κώδικα

1.1:
 - Yλοποίηση του πυρήνα `find_nearest_cluster` :
 ```c 
  __global__ static
void find_nearest_cluster()
{

	/* Get the global ID of the thread. */
    int tid = get_tid();

    if (tid < numObjs) {
        int   index, i;
        double dist, min_dist;

        /* find the cluster id that has min distance to object */
        index = 0;
        min_dist = euclid_dist_2(numCoords, numObjs, numClusters, objects, deviceClusters, tid, 0);

        for (i=1; i<numClusters; i++) {
            dist = euclid_dist_2(numCoords, numObjs, numClusters, objects, deviceClusters, tid, i);

            /* no need square root */
            if (dist < min_dist) { 
            /* find the min and its array index*/
                min_dist = dist;
                index    = i;
            }
        }

        if (deviceMembership[tid] != index) {
            atomicAdd(devdelta, 1.0);
        }

        /* assign the deviceMembership to object objectId */
        deviceMembership[tid] = index;
    }}
  ```
  - Yλοποίηση υπορουτίνας `euclid_dist_2` :
 ```c 
  __host__ __device__ inline static
double euclid_dist_2()
{
    int i;
    double ans=0.0;

    for (i=0; i<numCoords; i++)
        ans += (objects[objectId*numCoords + i] - clusters[clusterId*numCoords + i]) *
               (objects[objectId*numCoords + i] - clusters[clusterId*numCoords + i]);

    return(ans);
}
  ```
   - Υλοποίηση υπορουτίνας `get_tid`:
  ```c 
  __device__ int get_tid(){
	return blockDim.x * blockIdx.x + threadIdx.x;
}
  ```

1.2:
  - Πραγματοποιούμε  τις ζητούμενες μεταφορές δεδομένων σε κάθε iteration του αλγορίθμου:
   ```c 
   
checkCuda(cudaMemcpy(deviceClusters, clusters,
numClusters*numCoords*sizeof(double), cudaMemcpyHostToDevice));

checkCuda(cudaMemset(dev_delta_ptr, 0, sizeof(double)));

		
cudaDeviceSynchronize(); checkLastCudaError();
		
checkCuda(cudaMemcpy(membership, deviceMembership,
    numObjs*sizeof(int), cudaMemcpyDeviceToHost));

checkCuda(cudaMemcpy(&delta, dev_delta_ptr,
sizeof(double), cudaMemcpyDeviceToHost));
  ```



2.Αξιολόγηση επίδοσης:
 
 2.1 Μετρήσεις:
- Configuration:
    - Size = 256
    - Coords = 2
    - Clusters = 16
    - Loops = 10

   <insert plots and comments >


2.2 Σχολιασμός:

   <insert  comments >


### Υλοποίηση Transpose version

1.Υλοποίηση κώδικα

1.1:
 - Yλοποίηση του πυρήνα `euclid_dist_2_transpose` :
 ```c 
  __host__ __device__ inline static
double euclid_dist_2_transpose()
{
    int i;
    double ans=0.0;

    for (i=0; i<numCoords; i++) {
        ans += (objects[i*numObjs + objectId] - clusters[i*numClusters + clusterId]) *
               (objects[i*numObjs + objectId] - clusters[i*numClusters + clusterId]);
    }

    return(ans);
}
  ```
1.2:
 - Φροντίζουμε επίσης για την σωστή αρχικοποίηση και μετατροπή των δεδομένων:
  ```c 
double  **dimObjects = (double**) calloc_2d (numCoords, numObjs, sizeof(double)); 
   
double  **dimClusters = (double**) calloc_2d (numCoords, numClusters, sizeof(double)); 
   
double  **newClusters = (double**) calloc_2d (numCoords, numClusters, sizeof(double)); 
  ```

  2.Αξιολόγηση επίδοσης:
 
 2.1 Μετρήσεις:
- Configuration:
    - Size = 256
    - Coords = 2
    - Clusters = 16
    - Loops = 10

   <insert plots and comments >

2.2 Σχολιασμός διαφοράς επίδοσης: 


### Υλοποίηση Shared version

1.Υλοποίηση κώδικα

1.1:
 - Ορίζουμε το μέγεθος της διαμοιραζόμενης μνήμης που χρειάζεται η συγκεκριμένη υλοποιήση:
 ```c 
const unsigned int clusterBlockSharedDataSize = numClusters * numCoords * sizeof(double);
```
1.2:
 - Προσθέτουμε στον πυρήνα `find_nearest_cluster` την μεταφορά των clusters στην διαμοιραζόμενη μνήμη:
  ```c 
  if (tid < numClusters*numCoords) {
        shmemClusters[tid] = deviceClusters[tid];
    }
  ```


2. Μετρήσεις και αξιολόγηση
- Configuration:
    - Size = 256
    - Coords = 2
    - Clusters = 16
    - Loops = 10


### Σύγκριση υλοποίησεων / bottleneck Analysis
1. <insert comments>
2. 
- Configuration:
    - Size = 256
    - Coords = 16
    - Clusters = 16
    - Loops = 10
<insert comments>



  





 
 


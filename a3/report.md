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
1. Υλοποίηση κώδικα
  - Υλοποίηση υπορουτίνας `get_tid()`:
```c
__device__ int get_tid()
{
  return blockDim.x * blockIdx.x + threadIdx.x;
}
```
  - Yλοποίηση υπορουτίνας `euclid_dist_2()` :
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
  - Yλοποίηση του πυρήνα `find_nearest_cluster()` :
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

  - Πραγματοποιούμε  τις ζητούμενες μεταφορές δεδομένων σε κάθε iteration του αλγορίθμου:
```c
checkCuda(cudaMemcpy(deviceClusters, clusters, numClusters*numCoords*sizeof(double), cudaMemcpyHostToDevice));

/* ... */

checkCuda(cudaMemcpy(membership, deviceMembership, numObjs*sizeof(int), cudaMemcpyDeviceToHost));

checkCuda(cudaMemcpy(&delta, dev_delta_ptr, sizeof(double), cudaMemcpyDeviceToHost));
```



2. Αξιολόγηση επίδοσης:

  - Γραφικές Παραστάσεις
<img src='kmeans/plots/naive_2_16.png'>
<img src='kmeans/plots/naive_2_16_speedup.png'>

Παρατηρούμε μια μεγάλη βελτίωση στους χρόνους εκτέλεσης της παράλλληλης έκδοσης σε σχέση με την σειριακή. Συγκεκριμένα για ολα τα block sizes, το speedup είναι περισσότερο απο 10. H βελτίωση αυτή επιδεικνύει την αποτελεσματική χρήση των πόρων της GPU.

Για αυτό το configuration βλέπουμε ότι η διαφορά στους χρόνους για  block sizes εύρους 32-1024 είναι αμελητεά. Αυτό μπορεί να οφείλεται σε μερικούς λόγους, όπως:
1. Οι σύγχρονες κάρτες γραφικών είναι σχεδιασμένες ώστε να είναι σε θέση να μπορούν να διαχειριστούν με ευκολία αυτό το εύρος.
2. Η naive έκδοση του αλγόριθμου είναι memory-bound, δηλαδή η επίδοση του  περιορίζεται από τον ρυθμό με τον οποίον μεταφέρονται δεδομένα μεταξύ της κεντρικής μνήμης και των υπολογισικών νημάτων, παρά απο τον υπολογισμό τον ίδιο.
3. Ο αλγόριθμος σε ορισμένα σημεία χρησιμοποιεί ατομικές εντολές για την ενήμερωση δεδομένων με αποτέλεσμα να σειριοποιέιται η πρόσβαση σε αυτά και έτσι να μειώνονται τα οφέλη τα οποία προσδίδει ο μεγαλύτερος αριθμός νημάτων για κάθε block.


### Υλοποίηση Transpose version

1. Υλοποίηση κώδικα

  - Yλοποίηση του πυρήνα `euclid_dist_2_transpose()` :
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

  - Φροντίζουμε επίσης για την σωστή αρχικοποίηση και μετατροπή των δεδομένων:
```c
double  **dimObjects = (double**) calloc_2d (numCoords, numObjs, sizeof(double));

double  **dimClusters = (double**) calloc_2d (numCoords, numClusters, sizeof(double));

double  **newClusters = (double**) calloc_2d (numCoords, numClusters, sizeof(double));

/* ... */

for(i=0; i<numObjs; i++) {
  for(j=0; j<numCoords; j++) {
    dimObjects[j][i] = objects[i*numCoords + j];
  }
}

/* ... */

const unsigned int numClusterBlocks = (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;

/* ... */

for(i=0; i<numClusters; i++) {
  for(j=0; j<numCoords; j++) {
    clusters[i*numCoords + j] = dimClusters[j][i];
  }
}
```

2. Αξιολόγηση επίδοσης:
  - Γραφικές παραστάσεις
<img src='kmeans/plots/naive_transpose_2_16.png'>
<img src='kmeans/plots/naive_transpose_2_16_speedup.png'>

Παρατηρούμε ότι η transpose έκδοση του αλγορίθμου παρουσίαζει μια μικρή βελτίωση στην επίδοση σε σχέση με την naive έκδοση. Παρόλα αυτά, βλέπουμε ξανά ότι το block size δεν παίζει κάποιο ιδιαίτερο ρόλο.

Η αύξηση της επίδοσης της transpose έκδοσης πιθανώς οφείλεται στη διαφορετική δομή δεδομένων και στο μοτίβο με τον οποίο ανακτώνται τα αυτά τα δεδομένα απο την μνήμη. Στην column-based διάταξη, οι γραμμές των πινάκων αντιπροσωπεύουν μια διάσταση, ενώ οι στύλες κάποιο cluster center. Αυτό σημαίνει ότι οι συντεταγμένες των clusters αποθηκεύονται με συνεχή τρόπο, και συνεπώς η πρόσβαση σε αυτές γίνεται παρομοίως. Με αυτόν τον τρόπο, μειώνουμε σε κάποιο βαθμό το memory latency που παρουσιάζεται με την naive έκδοση, και ενισχύουμε την υπολογιστική αποδοτικότητα.

### Υλοποίηση Shared version

1. Υλοποίηση κώδικα

  - Ορίζουμε το μέγεθος της διαμοιραζόμενης μνήμης που χρειάζεται η συγκεκριμένη υλοποιήση:
```c
const unsigned int clusterBlockSharedDataSize = numClusters * numCoords * sizeof(double);
```
  - Προσθέτουμε στον πυρήνα `find_nearest_cluster()` την μεταφορά των clusters στην διαμοιραζόμενη μνήμη:
```c
if (tid < numClusters*numCoords) {
  shmemClusters[tid] = deviceClusters[tid];
}

__syncthreads();
```


2. Αξιολόγηση επίδοσης
  - Γραφικες Παραστάσεις
<img src='kmeans/plots/naive_transpose_shared_2_16.png'>
<img src='kmeans/plots/naive_transpose_shared_2_16_speedup.png'>

Παρατηρούμε ότι, σε σύγκριση με τις naive και transpose εκδόσεις του αλγορίθμου, η shared έκδοση έχει παρόμοια επίδοση με τις δυο προηγούμενες, κάνοντας όμως διαφορά στα μεγαλύτερα block sizes. Η ταχύτητά της για block sizes 512-1024 είναι αρκετά καλύτερη.

Σε αυτήν την περίπτωση, όταν το block size είναι τόσο μεγάλο, όλο και περισσότερα νήματα της κάρτας γραφικών μπορούν να έχουν πρόσβαση στα δεδομένα ενός block. Με αυτόν τον τρόπο αξιοποιείται καλύτερα η κοινή μνήμη, επειδή επιτυγχάνουμε λιγότερες συνολικά προσβάσεις στην κύρια μνήμη, αποφεύγουμε το κοστός που επιφέρουν, και έχουμε περισσότερο φόρτο εργασίας για τις υπολογιστικές μονάδες τις κάρτας γραφικών.


### Σύγκριση υλοποίησεων / bottleneck Analysis
  - Γραφικές παραστάσεις
<img src='kmeans/plots/naive_transpose_shared_16_16.png'>
<img src='kmeans/plots/naive_transpose_shared_16_16_speedup.png'>

Παρατηρούμε ότι η

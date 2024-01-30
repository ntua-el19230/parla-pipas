import re
import json
import sys

def parse_data(input_data):
    # Regular expression patterns
    dataset_pattern = r"dataset_size = (\d+.\d+) MB\s+numObjs = (\d+)\s+numCoords = (\d+)\s+numClusters = (\d+), block_size = (\d+)"
    seq_kmeans_pattern = r"Sequential Kmeans.*?total = ([\d.]+) ms"
    gpu_kmeans_pattern = r"(\w+) GPU Kmeans.*?t_alloc: ([\d.]+) ms.*?t_alloc_gpu: ([\d.]+) ms.*?t_get_gpu: ([\d.]+) ms.*?nloops = (\d+)\s+: total = ([\d.]+) ms.*?t_loop_avg = ([\d.]+) ms.*?t_loop_min = ([\d.]+) ms.*?t_loop_max = ([\d.]+) ms"

    results = []

    # Splitting the data into sections for each dataset
    datasets = input_data.split("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")[1:]

    for dataset in datasets:
        # Extracting dataset info
        dataset_info = re.search(dataset_pattern, dataset)
        if dataset_info:
            size, num_objs, num_coords, num_clusters, block_size = dataset_info.groups()
            size = int(float(size))
            num_coords = int(num_coords)
            num_clusters = int(num_clusters)
            block_size = int(block_size)

        # Extracting sequential Kmeans info
        seq_time = re.search(seq_kmeans_pattern, dataset, re.DOTALL)
        seqtime = float(seq_time.group(1)) if seq_time else None

        # Extracting GPU Kmeans info
        gpu_kmeans = re.search(gpu_kmeans_pattern, dataset, re.DOTALL)
        if gpu_kmeans:
            gpu_impl, t_alloc, t_alloc_gpu, t_get_gpu, nloops, time, t_loop_avg, t_loop_min, t_loop_max = gpu_kmeans.groups()

            # Creating a dictionary for each dataset
            result = {
                "implementation": gpu_impl.lower(),
                "size": size,
                "coords": num_coords,
                "clusters": num_clusters,
                "loops": int(nloops),
                "block_size": block_size,
                "seqtime": seqtime,
                "time": float(time),
                "t_alloc": float(t_alloc),
                "t_alloc_gpu": float(t_alloc_gpu),
                "t_get_gpu": float(t_get_gpu),
                "t_loop_avg": float(t_loop_avg),
                "t_loop_min": float(t_loop_min),
                "t_loop_max": float(t_loop_max)
            }
            results.append(result)

    return json.dumps(results, indent=2)

filename = sys.argv[1]
try:
  with open(filename, "r") as f:
    input_data = f.read()
  print(parse_data(input_data))

except FileNotFoundError:
  print("File not found.", file=sys.stderr)

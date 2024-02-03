import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

processors = [1, 2, 4, 8, 16, 32, 64]

def time_plot(data, filename, title):
  size = "256"
  coords = "16"
  clusters = "16"
  loops = "10"

  fig, ax = plt.subplots(figsize=(12, 8))
  bars = len(data)
  barwidth = 0.5

  times = [benchmark['time'] for benchmark in data]
  plt.bar(np.arange(bars), times, barwidth)

  plt.xlabel('Processors', fontweight='bold', fontsize=14)
  plt.ylabel('Time (s)', fontweight='bold', fontsize=14)
  plt.suptitle(title, fontsize=26)
  plt.title("{Size, Coords, Clusters, Loops} = " + "{" + size + ", " + coords + ", " + clusters + ", " + loops + "}")
  # plt.legend(title = "Processors")
  plt.grid(True)

  offset = 0
  plt.xticks([r + offset for r in range(bars)], processors)

  savepath = os.path.join('plots', filename)
  plt.savefig(savepath)
  plt.close()

def speedup_plot(data, filename, title):
  size = "256"
  coords = "16"
  clusters = "16"
  loops = "10"

  fig, ax = plt.subplots(figsize=(12, 8))
  bars = len(data)
  barwidth = 0.5

  seqtime = data[0]['time']
  times = [benchmark['time'] for benchmark in data]
  speedups = [seqtime / times[i] for i in range(len(times))]
  plt.bar(np.arange(bars), speedups, barwidth)

  plt.xlabel('Processors', fontweight='bold', fontsize=14)
  plt.ylabel('Speedup', fontweight='bold', fontsize=14)
  plt.suptitle(title, fontsize=26)
  plt.title("{Size, Coords, Clusters, Loops} = " + "{" + size + ", " + coords + ", " + clusters + ", " + loops + "}")
  # plt.legend(title = "Processors")
  plt.grid(True)

  offset = 0
  plt.xticks([r + offset for r in range(bars)], processors)

  savepath = os.path.join('plots', filename)
  plt.savefig(savepath)
  plt.close()

with open(sys.argv[1]) as f:
  data = json.load(f)

time_plot(data, 'kmeans_mpi_time.png', 'KMeans MPI Benchmark')
speedup_plot(data, 'kmeans_mpi_speedup.png', 'KMeans MPI Speedup')

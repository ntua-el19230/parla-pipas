import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

processors = [1, 2, 4, 8, 16, 32, 64]

def speedup_plot(data, filename, title):
  fig, ax = plt.subplots(figsize=(12, 8))
  bars = len(data)
  barwidth = 0.5

  seqtime = data[0]['time']
  times = [benchmark['time'] for benchmark in data]
  speedups = [seqtime / times[i] for i in range(len(times))]
  plt.bar(np.arange(bars), speedups, barwidth, label='Jacobi')

  plt.xlabel('Processors', fontweight='bold', fontsize=14)
  plt.ylabel('Speedup', fontweight='bold', fontsize=14)
  plt.suptitle(title, fontsize=26)
  plt.title("Jacobi method")
  plt.legend(title = "Array Size")
  plt.grid(True)

  offset = 0
  plt.xticks([r + offset for r in range(bars)], processors)

  savepath = os.path.join('plots', filename)
  plt.savefig(savepath)
  plt.close()

with open(sys.argv[1]) as f:
  data = json.load(f)

speedup_plot(data, 'heat_mpi_speedup.png', 'Heat Transfer MPI Speedup')

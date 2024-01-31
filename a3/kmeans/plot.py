import json
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

bsizes = [32, 64, 128, 256, 512, 1024]
width = 0.5

def speedup_plot(data, filename, title):
  size = str(data[0]['size'])
  coords = str(data[0]['coords'])
  clusters = str(data[0]['clusters'])
  loops = str(data[0]['loops'])

  implementations = set()
  for benchmark in data:
    implementations.add(benchmark['implementation'])

  fig, ax = plt.subplots(figsize=(12, 8))
  bars = len(implementations)
  barwidth = width / bars

  br = np.arange(6)
  for impl in implementations:
    seqtimes = [benchmark['seqtime'] for benchmark in data if benchmark['implementation'] == impl]
    times = [benchmark['time'] for benchmark in data if benchmark['implementation'] == impl]
    speedups = [seqtimes[i] / times[i] for i in range(len(times))]
    plt.bar(br, speedups, barwidth, label=impl.capitalize())
    br = [x + barwidth for x in br]

  plt.xlabel('Block Size', fontweight='bold', fontsize=14)
  plt.ylabel('Speedup', fontweight='bold', fontsize=14)
  plt.suptitle(title, fontsize=26)
  plt.title("{Size, Coords, Clusters, Loops} = " + "{" + size + ", " + coords + ", " + clusters + ", " + loops + "}")
  plt.legend(title = "Implementation")
  plt.grid(True)

  offset = barwidth * (bars - 1) / 2
  plt.xticks([r + offset for r in range(6)], bsizes)

  savepath = os.path.join('plots', filename)
  plt.savefig(savepath)
  plt.close()

def time_plot(data, filename, title):
  size = str(data[0]['size'])
  coords = str(data[0]['coords'])
  clusters = str(data[0]['clusters'])
  loops = str(data[0]['loops'])

  # find the distinct number of implementations in data
  implementations = set()
  for benchmark in data:
    implementations.add(benchmark['implementation'])

  fig, ax = plt.subplots(figsize=(12, 8))
  bars = len(implementations) + 1
  barwidth = width / bars

  br = np.arange(6)
  seqtimes = [benchmark['seqtime'] for benchmark in data[:6]]
  plt.bar(br, seqtimes, barwidth, label="Sequential")
  for impl in implementations:
    br = [x + barwidth for x in br]
    times = [benchmark['time'] for benchmark in data if benchmark['implementation'] == impl]
    plt.bar(br, times, barwidth, label=impl.capitalize())

  plt.xlabel('Block Size', fontweight='bold', fontsize=14)
  plt.ylabel('Time (ms)', fontweight='bold', fontsize=14)
  plt.suptitle(title, fontsize=26)
  plt.title("{Size, Coords, Clusters, Loops} = " + "{" + size + ", " + coords + ", " + clusters + ", " + loops + "}")
  plt.legend(title = "Implementation")
  plt.grid(True)

  offset = barwidth * (bars - 1) / 2
  plt.xticks([r + offset for r in range(6)], bsizes)

  savepath = os.path.join('plots', filename)
  plt.savefig(savepath)
  plt.close()

with open(sys.argv[1], 'r') as file:
  data = json.load(file)

naive = list(filter(lambda x: x['implementation'] == 'naive' and x['coords'] == 2 and x['clusters'] == 16, data))
transpose = list(filter(lambda x: x['implementation'] == 'transpose' and x['coords'] == 2 and x['clusters'] == 16, data))
shared = list(filter(lambda x: x['implementation'] == 'shared' and x['coords'] == 2 and x['clusters'] == 16, data))

naive_transpose = list(filter(lambda x: (x['implementation'] == 'naive' or x['implementation'] == 'transpose') and x['coords'] == 2 and x['clusters'] == 16, data))
naive_transpose_shared = list(filter(lambda x: x['coords'] == 2 and x['clusters'] == 16, data))
naive_transpose_shared2 = list(filter(lambda x: x['coords'] == 16 and x['clusters'] == 16, data))

time_plot(naive, 'naive_2_16.png', 'Naive GPU K-Means')
speedup_plot(naive, 'naive_2_16_speedup.png', 'Naive GPU K-Means Speedup')

time_plot(naive_transpose, 'naive_transpose_2_16.png', 'Naive vs Transpose GPU K-Means')
speedup_plot(naive_transpose, 'naive_transpose_2_16_speedup.png', 'Naive vs Transpose GPU K-Means Speedup')

time_plot(naive_transpose_shared, 'naive_transpose_shared_2_16.png', 'Naive vs Transpose vs Shared GPU K-Means')
speedup_plot(naive_transpose_shared, 'naive_transpose_shared_2_16_speedup.png', 'Naive vs Transpose vs Shared GPU K-Means Speedup')

time_plot(transpose, 'transpose_2_16.png', 'Transpose GPU K-Means')
speedup_plot(transpose, 'transpose_2_16_speedup.png', 'Transpose GPU K-Means Speedup')

time_plot(shared, 'shared_2_16.png', 'Shared GPU K-Means')
speedup_plot(shared, 'shared_2_16_speedup.png', 'Shared GPU K-Means Speedup')

time_plot(naive_transpose_shared2, 'naive_transpose_shared_16_16.png', 'Naive vs Transpose vs Shared GPU K-Means')
speedup_plot(naive_transpose_shared2, 'naive_transpose_shared_16_16_speedup.png', 'Naive vs Transpose vs Shared GPU K-Means Speedup')

print("Done!")

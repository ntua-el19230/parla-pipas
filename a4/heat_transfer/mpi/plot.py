import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

processors = [1, 2, 4, 8, 16, 32, 64]
width = 0.5

def speedup_plot(data, filename, title):
  fig, ax = plt.subplots(figsize=(12, 8))
  sizes = set(benchmark['size'] for benchmark in data)
  bars = len(sizes)
  barwidth = width / bars

  br = np.arange(7)
  for size in sizes:
    benchmarks = [benchmark for benchmark in data if benchmark['size'] == size]
    benchmarks.sort(key=lambda x: x['threads'])
    times = [benchmark['total'] for benchmark in benchmarks]
    seqtime = times[0]
    speedups = [seqtime / time for time in times]
    plt.bar(br, speedups, barwidth, label=f'{size}x{size}')
    br = [x + barwidth for x in br]

  plt.xlabel('Processors', fontweight='bold', fontsize=14)
  plt.ylabel('Speedup', fontweight='bold', fontsize=14)
  plt.suptitle(title, fontsize=26)
  plt.title("Jacobi method")
  plt.legend(title = "Array Size")
  plt.grid(True)

  offset = barwidth * (bars - 1) / 2
  plt.xticks([r + offset for r in range(len(processors))], processors)

  savepath = os.path.join('plots', filename)
  plt.savefig(savepath)
  plt.close()

def time_plot(data, filename, title):
  fig, ax = plt.subplots(figsize=(12, 8))
  bars = 2
  barwidth = width / bars

  data = sorted(data, key=lambda x: x['threads'])

  br = np.arange(7)
  comp_times = [benchmark['comp'] for benchmark in data]
  plt.bar(br, comp_times, barwidth, label='Computation Time')

  br = [x + barwidth for x in br]
  total_times = [benchmark['total'] for benchmark in data]
  plt.bar(br, total_times, barwidth, label='Total Time')

  plt.xlabel('Processors', fontweight='bold', fontsize=14)
  plt.ylabel('Time (s)', fontweight='bold', fontsize=14)
  plt.suptitle(title, fontsize=26)
  plt.title("Jacobi method")
  plt.legend(title = "Time Range")
  plt.grid(True)

  offset = barwidth * (bars - 1) / 2
  plt.xticks([r + offset for r in range(len(processors))], processors)

  savepath = os.path.join('plots', filename)
  plt.savefig(savepath)
  plt.close()

def average_benchmarks(data):
  averages = []
  for i in range(len(data[0])):
    size = data[0][i]['x']
    threads = data[0][i]['px'] * data[0][i]['py']
    average_computation = sum(benchmarks[i]['comp_time'] for benchmarks in data) / len(data)
    average_total = sum(benchmarks[i]['total_time'] for benchmarks in data) / len(data)

    averages.append({
      'size': size,
      'threads': threads,
      'comp': average_computation,
      'total': average_total
    })

  return averages

data = []

for i in range(1, 4):
  filename = f'benchmarks{i}.json'
  with open(filename, 'r') as file:
    benchmarks = json.load(file)
    data.append(benchmarks)

data = average_benchmarks(data)

speedup_plot(data, 'heat_mpi_speedup.png', 'Heat Transfer MPI Speedup')

for size in set(benchmark['size'] for benchmark in data):
  benchmarks = [benchmark for benchmark in data if benchmark['size'] == size]
  time_plot(benchmarks, f'heat_mpi_time_{size}x{size}.png', f'Heat Transfer MPI Time {size}x{size}')

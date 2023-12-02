import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np

width = 0.15
implementations = [
  'Coarse-Grained Locking',
  'Fine-Grained Locking',
  'Optimistic Synchronization',
  'Lazy Synchronization',
  'Non-Blocking Synchronization'
]

def plot(tests, filename, legend, title):
  fig, ax = plt.subplots(figsize=(12, 8))
  br = np.arange(8)
  size = str(tests[0]['Size'])
  workload = tests[0]['Workload']

  for implementation in implementations:
    subtests = list(filter(lambda x: x['Implementation'] == implementation, tests))
    subtests = sorted(subtests, key=lambda x: x['Nthreads'])
    plot_util(subtests, implementation, br)
    br = [x + width for x in br]

  plt.xlabel('Threads', fontweight='bold', fontsize=14)
  plt.ylabel('Throughput', fontweight='bold', fontsize=14)
  plt.suptitle(title)
  plt.title("Size: " + size + " Workload: " + workload)
  plt.legend(title = legend)
  plt.grid(True)
  plt.xticks([r + 0.3 for r in range(8)], ['1', '2', '4', '8', '16', '32', '64', '128'])

  savepath = os.path.join('plots', filename)
  plt.savefig(savepath)
  plt.close()

def plot_util(measurements, label, offset):
  throughput = []
  for ms in measurements:
    throughput.append(ms['Throughput'])

  plt.bar(offset, throughput, width, label=f"{label}")

def str_wl(wl):
  return wl.replace('/', '-')

with open('performance.json', 'r') as file:
  data = json.load(file)

for size in [1024, 8192]:
  for workload in ['100/0/0', '80/10/10', '20/40/40', '0/50/50']:
    # plot((filter(lambda x: x['Size'] == size and x['Workload'] == workload, data), f'conc_ll_{size}_{str_wl(workload)}.png', 'Implementation', f'Concurrent Linked Lists'))
    plot(list(filter(lambda x: x['Size'] == size and x['Workload'] == workload, data)), f'conc_ll_{size}_{str_wl(workload)}.png', 'Implementation', f'Concurrent Linked Lists')


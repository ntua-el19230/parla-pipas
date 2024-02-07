import re
import json
import sys

def parse_data(input_data):
  pattern = r"Jacobi X (\d+) Y (\d+) Px (\d+) Py (\d+) Iter (\d+) ComputationTime (\d+.\d+) TotalTime (\d+.\d+) midpoint (\d+.\d+)"
  results = []

  for line in input_data.split("\n"):
    match = re.match(pattern, line)
    if match:
      x, y, px, py, iterations, comp_time, total_time, midpoint = match.groups()
      result = {
        "x": int(x),
        "y": int(y),
        "px": int(px),
        "py": int(py),
        "iter": int(iterations),
        "comp_time": float(comp_time),
        "total_time": float(total_time),
        "midpoint": float(midpoint)
      }
      results.append(result)

  return json.dumps(results, indent=2)

with open(sys.argv[1], "r") as f:
  input_data = f.read()
print(parse_data(input_data))

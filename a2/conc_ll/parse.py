import re
import json
import sys

def parse_input(file_name):
  implementations = []
  with open(file_name, 'r') as file:
    data = file.read()

  pattern = r'Implementation:\s*(.*?)\n\s*MT_CONF=\s*(.*?)\n\s*Nthreads:\s*(\d+)\s*Runtime\(sec\):\s*(\d+)\s*Size:\s*(\d+)\s*Workload:\s*(\d+/\d+/\d+)\s*Throughput\(Kops\/sec\):\s*([\d.]+)\s*'

  matches = re.findall(pattern, data, re.DOTALL)

  for match in matches:
    implementation = {
      "Implementation": match[0],
      "MT_CONF": match[1],
      "Nthreads": int(match[2]),
      "Runtime": int(match[3]),
      "Size": int(match[4]),
      "Workload": match[5],
      "Throughput": float(match[6])
    }
    implementations.append(implementation)

  return implementations

file_name = sys.argv[1]
parsed_data = parse_input(file_name)

output_file_name = 'performance.json'
with open(output_file_name, 'w') as output_file:
  json.dump(parsed_data, output_file, indent=2)

print(f"Parsing completed. Result written to {output_file_name}")

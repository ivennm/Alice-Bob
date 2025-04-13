import numpy as np
import re

def parse_file(file_path, num_channels, num_samples):
    data_matrix = np.zeros((num_channels, num_samples))
    current_channel = -1
    current_sample = 0
    
    with open(file_path, 'r') as file:
        for line in file:
            # Check for headers
            if line.startswith("$ChnID"):
                channel_id = int(re.findall(r':(\d+)$', line)[0])
                current_channel = channel_id - 1  # Assuming channel IDs are 1-indexed
                current_sample = 0
            elif line.strip().isdigit() or (line.strip() and line.strip()[0] == '-'):
                # Process data line
                data_matrix[current_channel, current_sample] = int(line.strip())
                current_sample += 1
    
    return data_matrix

# -----------------------------------------

file_path = 'data_txt/wigner_fock1.txt'
num_channels = 5
num_samples = 10800
data_matrix = parse_file(file_path, num_channels, num_samples)

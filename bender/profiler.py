import pandas as pd
import re
import os

path_to_run = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bender_episode_4.py')
outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bender_episode_4_profile.log')
cmd_to_run = f'python -m cProfile \"{path_to_run}\" > \"{outfile}\"'
# Run command and save output to file
os.system(f'{cmd_to_run}')

# Read file
with open(outfile, 'r') as f:
    lines = f.readlines()
lines = lines[5:]
s = ''.join(lines)

st = s.split('\n')
st = [stt for stt in st if stt]
cols = st[0].split()
st = st[1:]
st = [stt.replace('\\', '') for stt in st]

# Using regex, find an integer, followed by a float, followed by a float, followed by a float, followed by a float followed by a string
pattern = re.compile(r'(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(.*)')
st = [pattern.findall(stt) for stt in st]
st = [stt for stt in st if stt]
st = [(stt[0][0], float(stt[0][1]), float(stt[0][2]), float(stt[0][3]), float(stt[0][4]), stt[0][5]) for stt in st]
df = pd.DataFrame(st, columns=cols)
df = df.sort_values(by='tottime', ascending=False)

# Plot the top 10 functions
xdf = df.head(10)
# pretty print xdf
print(xdf.to_string(index=False))
j = 3

import sys
import math
import re


# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

def dprint(s=''):
    print(s, file=sys.stderr)


def get_val(arg, d):
    if arg == '_':
        return -999
    elif '$' in arg:
        return d.get(int(arg[1:]), None)
    else:
        return int(arg)


n = int(input())
dprint(f'Lines: {n}')
l = list()
d_mem = dict()
for i in range(n):
    operation, arg_1, arg_2 = input().split()
    dprint(f'[{i:>4}][{operation}]: {arg_1} <> {arg_2}')
    item = (i, operation, arg_1, arg_2)
    dprint(item)
    l += [item]

while len(l) > 0:
    lx = list()
    for (i, operation, arg_1, arg_2) in l:
        val_1 = get_val(arg_1, d_mem)
        val_2 = get_val(arg_2, d_mem)
        if (val_1 is None) or (val_2 is None):
            lx += [(i, operation, arg_1, arg_2)]
        else:
            if operation == 'VALUE':
                d_mem[i] = get_val(arg_1, d_mem)
            elif operation == 'ADD':
                d_mem[i] = get_val(arg_1, d_mem) + get_val(arg_2, d_mem)
            elif operation == 'SUB':
                d_mem[i] = get_val(arg_1, d_mem) - get_val(arg_2, d_mem)
            elif operation == 'MULT':
                d_mem[i] = get_val(arg_1, d_mem) * get_val(arg_2, d_mem)
            else:
                dprint(f"ERROR, operation: {operation}")
    l = lx

for i in range(n):
    print(d_mem[i])

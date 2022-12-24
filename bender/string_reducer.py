import numpy as np
import pandas as pd
from copy import copy
from collections import defaultdict

s = 'RRRRDDRRDDDDDDRLULLDDDRRDDDDRRDDRRRRRRRRLLLLLLRLLLUULLUUUULLUUULLUDRRURRUUUULLUULLLLUURRRRRRRLRRRRRRDDDDRRRDRDDDLLUDRRDDUDDLLDLLRRURRUUUDUUUUULLUULLUULLLLLRLLLLLLLLDDDDLRDDDDDDRRURRRRDRLRRRRRRUUU'

sq = s


def get_value(s, max_pattern_length=8):
    def find_most_repeating_pattern(s: str, k: int) -> str:
        # Create a dictionary to store the count of each pattern
        # Keep track of the frequency of each pattern
        pattern_counts = defaultdict(int)

        # Initialize the sliding window
        window_start = 0
        window_end = k

        # Slide the window over the string, counting the frequency of each pattern
        while window_end <= len(s):
            pattern = s[window_start:window_end]
            pattern_counts[pattern] += 1
            window_start += 1
            window_end += 1

        # Find the pattern with the highest frequency
        most_common_pattern = max(pattern_counts, key=pattern_counts.get)
        repeats = pattern_counts[most_common_pattern]

        return most_common_pattern, repeats

    def find_best_string_to_reduce(s, max_pattern_length=8):
        hit = False
        best_pattern = None
        best_value = 0
        max_pattern_length = min(max_pattern_length, len(s))

        if len(s) < 4:
            return None, None

        for pattern_length in range(3, max_pattern_length):
            pattern, repeats = find_most_repeating_pattern(s, pattern_length)
            value = (repeats * pattern_length) - repeats
            if value > best_value:
                best_value = value
                best_pattern = pattern
                if hit:
                    pass
                    # break
                else:
                    hit = True
        return best_pattern, value

    sq = copy(s)
    funcs = dict()
    possible_functions = 9
    for i in range(1, possible_functions + 1):
        best_pattern, best_repeats = find_best_string_to_reduce(sq, max_pattern_length)
        if best_pattern is None:
            break
        funcs[i] = best_pattern
        sq = sq.replace(best_pattern, f'{i}')

    resulting_string = ';'.join([sq] + [funcs[i] for i in range(1, len(funcs) + 1)])
    resulting_value = len(resulting_string)
    return resulting_string, resulting_value


def get_value_optimize(s):
    length_values = dict()
    j = 3
    for i in range(4, len(s) - 2):
        st, vt = get_value(s, i)
        length_values[i] = vt


resulting_string, resulting_value = get_value(s)
ls = resulting_string.split(';')
print(ls[0])
for i in range(1, len(ls)):
    print(f'F{i}: {ls[i]}')
print(f"<><><><> Score: {resulting_value}")

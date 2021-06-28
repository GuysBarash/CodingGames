import sys
import math
import numpy as np


def dprint(s=''):
    import sys
    print(s, file=sys.stderr)


# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

# nb_floors: number of floors
# width: width of the area
# nb_rounds: maximum number of rounds
# exit_floor: floor on which the exit is found
# exit_pos: position of the exit on its floor
# nb_total_clones: number of generated clones
# nb_additional_elevators: ignore (always zero)
# nb_elevators: number of elevators
#  0 nb_floors,
#  1 width
#  2 nb_rounds,
#  3 exit_floor,
#  4 exit_pos,
#  5 nb_total_clones,
#  6 nb_additional_elevators,
#  7 nb_elevators = [int(i) for i in input().split()]
a = [7, 24, 200, 6, 19, 40, 0, 6]  # [int(i) for i in input().split()]
qmap = [[2, 3], [5, 11], [1, 9], [4, 15], [0, 13], [3, 20]]  # [[int(j) for j in input().split()] for _ in range(a[7])]
qmap += [[a[3], a[4]]]
qmap = {q[0]: q[1] for q in qmap}

# game loop
while True:
    b = input().split()
    # clone_floor = 0 , int , b[0]
    # clone_pos = 11, int , b[1]
    # direction = 'RIGHT' int , b[2]

    if ((arr[int(b[0])].argmax() > int(b[1])) and b[2] == 'LEFT') or (
            (arr[int(b[0])].argmax() < int(b[1])) and b[2] == 'RIGHT'):
        print("BLOCK")
    else:
        print("WAIT")

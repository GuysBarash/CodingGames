import sys
import math

def dprint(s=''):
    print(s, file=sys.stderr)

factory_count = int(input())  # the number of factories
link_count = int(input())  # the number of links between factories
dprint(f"factories: {factory_count}\tRoads: {link_count}")
for i in range(link_count):
    factory_1, factory_2, distance = [int(j) for j in input().split()]
    dprint(f"[{i+2}/{link_count}]\t{factory_1} <--{distance}--> {factory_2}")

# game loop
while True:
    entity_count = int(input())  # the number of entities (e.g. factories and troops)
    for i in range(entity_count):
        inputs = input().split()
        entity_id = int(inputs[0])
        entity_type = inputs[1]
        arg_1 = int(inputs[2])
        arg_2 = int(inputs[3])
        arg_3 = int(inputs[4])
        arg_4 = int(inputs[5])
        arg_5 = int(inputs[6])

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)


    # Any valid action, such as "WAIT" or "MOVE source destination cyborgs"
    print("WAIT")

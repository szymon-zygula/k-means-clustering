#!/usr/bin/env python3

import sys
import random

if len(sys.argv) != 3:
    print("USAGE: ", sys.argv[0], " object_number dimension")
    exit(1)

count = int(sys.argv[1])
dim = int(sys.argv[2])

for i in range(count):
    for j in range(dim):
        print(random.uniform(-1.0, 1.0), end=" ")
    print()

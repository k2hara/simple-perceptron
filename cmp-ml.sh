#!/bin/sh
gcc -msse2 -DHAVE_SSE2 -DMEXP=1279 -o ./ml ./ml.c -lm

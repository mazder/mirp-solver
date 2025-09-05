#!/bin/bash

export LD_LIBRARY_PATH=/opt/gklib/lib:$LD_LIBRARY_PATH
export PATH=/opt/gklib/bin:/opt/gklib/include:$PATH
export LD_LIBRARY_PATH=/opt/metis5/lib:$LD_LIBRARY_PATH
export PATH=/opt/metis5/bin:/opt/metis5/include:$PATH
#ln -s /opt/gklib/lib/libGKlib.a /opt/gklib/lib/libgklib.a

grep IDXTYPEWIDTH /opt/metis5/include/metis.h
grep REALTYPEWIDTH /opt/metis5/include/metis.h


g++ -std=c++17 -I/opt/metis5/include main.cpp -L/opt/metis5/lib -lmetis -o test_metis -L/opt/gklib/lib -lgklib

./test_metis

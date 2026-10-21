OPENQASM 3.0;
include "stdgates.inc";
bit[6] c;
c[0] = measure $2;
c[1] = measure $4;
c[2] = measure $7;
c[3] = measure $5;
c[4] = measure $8;
c[5] = measure $11;

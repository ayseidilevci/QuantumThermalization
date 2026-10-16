OPENQASM 3.0;
include "stdgates.inc";
gate rzz(p0) _gate_q_0, _gate_q_1 {
  cx _gate_q_0, _gate_q_1;
  rz(p0) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
}
gate rxx(p0) _gate_q_0, _gate_q_1 {
  h _gate_q_0;
  h _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  rz(p0) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_1;
  h _gate_q_0;
}
qubit[6] q;
h q[3];
cx q[3], q[0];
h q[4];
cx q[4], q[1];
h q[5];
cx q[5], q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
rx(0.05097769445456119) q[0];
rx(0.05097769445456119) q[1];
rx(0.05097769445456119) q[2];
rx(0.05097769445456119) q[3];
rx(0.05097769445456119) q[4];
rx(0.05097769445456119) q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
rzz(-0.5207957044733019) q[0], q[1];
rzz(-0.5207957044733019) q[1], q[2];
rzz(-0.5207957044733019) q[3], q[4];
rzz(-0.5207957044733019) q[4], q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
rxx(0.7915463662537686) q[0], q[3];
rxx(0.7915463662537686) q[1], q[4];
rxx(0.7915463662537686) q[2], q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
rzz(-0.091385995873763) q[0], q[3];
rzz(-0.091385995873763) q[1], q[4];
rzz(-0.091385995873763) q[2], q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];

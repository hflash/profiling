//Bernstein-Vazirani with 8 qubits.
//Hidden string is 1111111
OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg cr[7];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
x q[7];
h q[7];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
cx q[0],q[7];
cx q[1],q[7];
cx q[2],q[7];
cx q[3],q[7];
cx q[4],q[7];
cx q[5],q[7];
cx q[6],q[7];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
measure q[0] -> cr[0];
measure q[1] -> cr[1];
measure q[2] -> cr[2];
measure q[3] -> cr[3];
measure q[4] -> cr[4];
measure q[5] -> cr[5];
measure q[6] -> cr[6];

//@author Raymond Harry Rudy rudyhar@jp.ibm.com
//Bernstein-Vazirani with 12 qubits.
//Hidden string is 11111111111
OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg cr[11];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
x q[11];
h q[11];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11];
cx q[0],q[11];
cx q[1],q[11];
cx q[2],q[11];
cx q[3],q[11];
cx q[4],q[11];
cx q[5],q[11];
cx q[6],q[11];
cx q[7],q[11];
cx q[8],q[11];
cx q[9],q[11];
cx q[10],q[11];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
measure q[0] -> cr[0];
measure q[1] -> cr[1];
measure q[2] -> cr[2];
measure q[3] -> cr[3];
measure q[4] -> cr[4];
measure q[5] -> cr[5];
measure q[6] -> cr[6];
measure q[7] -> cr[7];
measure q[8] -> cr[8];
measure q[9] -> cr[9];
measure q[10] -> cr[10];

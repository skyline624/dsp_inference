// Cluster-node build wrapper. Defines NODE_ONLY then textually includes top.v so
// the macro is guaranteed in scope (Gowin does not share `define across files
// added separately). With NODE_ONLY set, top.v disables the GG generation FSM
// (op_gg=0 + GG dispatch removed + GG state block compiled out) -> the dense GG
// monolith that caused the routing congestion is gone; only FN/FQ/MM/SS/EE/CN
// (the brick datapath the cluster uses) remain.
//
// The full design still builds from top.v directly (build.tcl), NODE_ONLY undefined.
`define NODE_ONLY
`include "top.v"

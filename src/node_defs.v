// Cluster-node build flag. When this file is compiled before top.v, NODE_ONLY
// is defined -> top.v disables the GG generation FSM (op_gg=0 + GG dispatch
// removed). The cluster node only needs FN/FQ/MM/SS/EE; removing the dense GG
// monolith relieves the routing congestion that blocked the full design.
`define NODE_ONLY

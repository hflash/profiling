# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 22:20:10 2019

@author: Xiangzhen Zhou
"""

from circuittransform import OperationU
# from circuittransform.machinelearning.data import CreateCircuitMapMultiLayer
from qiskit import QuantumRegister
import networkx as nx
import circuittransform as ct
import copy
import numpy as np

def OperationCost(dom, mapping, G = None, shortest_length = None, edges_DiG=None, shortest_path_G=None):
    '''
    calculate the cost(number of swaps) of input operation with its corresponding map in an unidrected architecture graph
    via the length of shortest path between 2 input qubits
    input:
        dom: an U operation or a list of its corresponding qubits or vertexes in architecture graph
    '''
    if isinstance(dom, OperationU):
        q0 = dom.involve_qubits[0]
        q1 = dom.involve_qubits[1]
        v0 = mapping.DomToCod(q0)
        v1 = mapping.DomToCod(q1)
    if isinstance(dom, list):
        # dom is qubits
        if isinstance(dom[0], tuple):
            q0 = dom[0]
            q1 = dom[1]
            v0 = mapping.DomToCod(q0)
            v1 = mapping.DomToCod(q1)
        # dom is vertexes
        else:
            v0 = dom[0]
            v1 = dom[1]
    
    if shortest_length != None:
        cost = shortest_length[v0][v1] - 1
    else:
        cost = nx.shortest_path_length(G, source=v0, target=v1, weight=None, method='dijkstra') - 1
    
    if edges_DiG != None:
        flag_4H = ct.CheckCNOTNeedConvertDirection(v0, v1, shortest_path_G[v0][v1], edges_DiG)
        cost += flag_4H * 4/7 #we only count the number of SWAP gates
    return cost

def HeuristicCostZulehner(current_map, DG, executable_vertex, shortest_length_G, shortest_path_G=None, DiG=None):
    '''
    Calculate heuristic cost for remaining gates
    see "An Efficient Methodology for Mapping Quantum Circuits to the IBM QX Architectures"
    '''
    worst_num_swap = None
    count_same_worst = 0
    sum_num_swap = 0
    best_num_swap = None
    mapping = current_map
    flag_finished = True
    if DiG != None: edges = list(DiG.edges)
    for v_DG in executable_vertex:
        current_operation = DG.nodes[v_DG]['operation']
        q0 = current_operation.involve_qubits[0]
        q1 = current_operation.involve_qubits[1]
        v0 = mapping.DomToCod(q0)
        v1 = mapping.DomToCod(q1)
        current_num_swap = shortest_length_G[v0][v1] - 1
        if current_num_swap > 0: flag_finished = False
        '''if architecture graph is directed, confirm whether use 4 H gates to change direction'''
        if DiG != None:
            flag_4H = ct.CheckCNOTNeedConvertDirection(v0, v1, shortest_path_G[v0][v1], edges)
            #print('flag_4H is', flag_4H)
            current_num_swap += flag_4H * 4/7 #we only count the number of SWAP gates
        '''renew number of all swaps'''
        sum_num_swap = sum_num_swap + current_num_swap
        '''renew swap number of worst operation'''
        if worst_num_swap == None:
            worst_num_swap = current_num_swap
            worst_vertex = v_DG
        else:
            if current_num_swap > worst_num_swap:
                worst_num_swap = current_num_swap
                worst_vertex = v_DG
                count_same_worst = 0
            else:
                if current_num_swap == worst_num_swap:
                    count_same_worst += 1
        '''renew swap number of best operation'''
        if best_num_swap == None:
            best_num_swap = current_num_swap
        else:
            if current_num_swap < best_num_swap:
                best_num_swap = current_num_swap
        
# =============================================================================
# worst_num_swap += sum_num_swap/100 #when identical worst costs for different
# gates exist, consider sum of swaps as second goal
# =============================================================================
    return worst_num_swap, sum_num_swap, best_num_swap, count_same_worst, worst_vertex, flag_finished

def HeuristicCostZulehnerLookAhead(current_map, DG, executable_vertex, shortest_length_G, shortest_path_G=None, DiG=None):
    '''
    Calculate heuristic cost for remaining gates
    see "An Efficient Methodology for Mapping Quantum Circuits to the IBM QX Architectures"
    '''
    sum_num_swap = 0
    current_H_num = 0
    mapping = current_map
    finished = False
    if DiG != None: edges = list(DiG.edges)
    '''calculate cost for current level'''
    for v_DG in executable_vertex:
        current_operation = DG.nodes[v_DG]['operation']
        q0 = current_operation.involve_qubits[0]
        q1 = current_operation.involve_qubits[1]
        v0 = mapping.DomToCod(q0)
        v1 = mapping.DomToCod(q1)
        current_num_swap = shortest_length_G[v0][v1] - 1
        '''if architecture graph is directed, confirm whether use 4 H gates to change direction'''
        if DiG != None:
            flag_4H = ct.CheckCNOTNeedConvertDirection(v0, v1, shortest_path_G[v0][v1], edges)
            current_H_num += flag_4H * 4/7 #we only count the number of SWAP gates
        '''renew number of all swaps'''
        sum_num_swap = sum_num_swap + current_num_swap
    current_level_num_swap = sum_num_swap
    if current_level_num_swap == 0: finished = True
    current_level_num_swap += current_H_num
    sum_num_swap += current_H_num
    '''calculate cost for next level'''
    DG_copy = DG.copy()
    DG_copy.remove_nodes_from(executable_vertex)
    lookahead_vertex = ct.FindExecutableNode(DG_copy)
    for v_DG in lookahead_vertex:
        current_operation = DG.nodes[v_DG]['operation']
        q0 = current_operation.involve_qubits[0]
        q1 = current_operation.involve_qubits[1]
        v0 = mapping.DomToCod(q0)
        v1 = mapping.DomToCod(q1)
        current_num_swap = shortest_length_G[v0][v1] - 1
        '''if architecture graph is directed, confirm whether use 4 H gates to change direction'''
        if DiG != None:
            flag_4H = ct.CheckCNOTNeedConvertDirection(v0, v1, shortest_path_G[v0][v1], edges)
            current_num_swap += flag_4H * 4/7
        '''renew number of all swaps'''
        sum_num_swap = sum_num_swap + current_num_swap
    
    return sum_num_swap, current_level_num_swap, finished

def HeuristicCostZhou1(current_map, DG, executed_vertex, executable_vertex,
                       shortest_length_G, shortest_path_G, level_lookahead,
                       DiG=None):
    '''
    Calculate heuristic cost for remaining gates and return best path
    this cost is based on the minimial distance in architecture graph between 
    two input qubits of each operations
    '''
    worst_num_swap = None
    count_same_worst = 0
    sum_num_swap = 0
    best_num_swap = None
    mapping = current_map
    best_executable_vertex = None
    best_path = None
    if DiG != None: edges = list(DiG.edges)
    #DG_copy = copy.deepcopy(DG)
    executable_vertex_copy = executable_vertex.copy()
    executed_vertex_copy = executed_vertex.copy()
    num_counted_gates = 0
    for current_lookahead_level in range(len(level_lookahead)):
        if current_lookahead_level == 0:
            '''current level'''
            current_executable_vertex = executable_vertex_copy
            weight = 1
        else:
            '''lookahead level'''
            #DG_copy.remove_nodes_from(executable_vertex)
            current_executable_vertex = ct.FindExecutableNode(DG, executed_vertex_copy, current_executable_vertex, current_executable_vertex.copy())
            weight = level_lookahead[current_lookahead_level - 1]
            
        num_counted_gates += len(current_executable_vertex)
        for v_DG in current_executable_vertex:
            flag_4H = 0
            current_operation = DG.nodes[v_DG]['operation']
            q0 = current_operation.involve_qubits[0]
            q1 = current_operation.involve_qubits[1]
            v0 = mapping.DomToCod(q0)
            v1 = mapping.DomToCod(q1)
            current_num_swap = shortest_length_G[v0][v1] - 1
            '''if architecture graph is directed, confirm whether use 4 H gates to change direction'''
            if DiG != None:
                flag_4H = ct.CheckCNOTNeedConvertDirection(v0, v1, shortest_path_G[v0][v1], edges)
            current_num_swap += flag_4H * 4/7
            '''renew number of all swaps'''
            current_num_swap = current_num_swap * weight#multiply the weight for cost of gates in different levels
            sum_num_swap = sum_num_swap + current_num_swap
            '''renew swap number of worst operation'''
            if worst_num_swap == None:
                worst_num_swap = current_num_swap
                count_same_worst = 0
            else:
                if current_num_swap > worst_num_swap:
                    worst_num_swap = current_num_swap
                    count_same_worst = 0
                else:
                    if current_num_swap == worst_num_swap:
                        count_same_worst += 1
            '''renew swap number of best operation'''
            if best_num_swap == None:
                best_num_swap = current_num_swap
                best_path = shortest_path_G[v0][v1]
                best_executable_vertex = v_DG
            else:
                if current_num_swap < best_num_swap:
                    best_num_swap = current_num_swap
                    best_path = shortest_path_G[v0][v1]
                    best_executable_vertex = v_DG
        
    return worst_num_swap, sum_num_swap, best_num_swap, best_executable_vertex, best_path, count_same_worst, num_counted_gates

def HeuristicCostZhouML(ANN, current_map, DG, executed_vertex,
                        executable_vertex, shortest_length_G, shortest_path_G,
                        level_lookahead, DiG=None):
    '''
    Calculate heuristic cost for remaining gates and return best path
    this cost is based on the minimial distance in architecture graph between
    two input qubits of each operations
    
    ATTENTION: this function is currently only for bidirectional AG Q20!!!
               for directional AG, it needs further modification
    
    input:
        ANN -> neural network via keras API
    '''
    mapping = current_map
    num_layer = len(level_lookahead)
    #DG_copy = copy.deepcopy(DG)
    executable_vertex_copy = executable_vertex.copy()
    executed_vertex_copy = executed_vertex.copy()
    current_executable_vertex = executable_vertex_copy
    num_counted_gates = 0
    data_set = np.zeros((num_layer, 1, 20, 20))
    num_q_log = 20
    weights = np.ones(num_layer)
    weights[1:] = level_lookahead[0:-1]
    
    for current_lookahead_level in range(num_layer):
        '''lookahead level'''
        #DG_copy.remove_nodes_from(executable_vertex)
        current_executable_vertex = ct.FindExecutableNode(DG,
                                                          executed_vertex_copy,
                                                          current_executable_vertex,
                                                          current_executable_vertex.copy())

        
        num_counted_gates += len(current_executable_vertex)
        CNOT_list = []
        for v_DG in current_executable_vertex:
            current_operation = DG.nodes[v_DG]['operation']
            q0 = current_operation.involve_qubits[0]
            q1 = current_operation.involve_qubits[1]
            v0 = mapping.DomToCod(q0)
            v1 = mapping.DomToCod(q1)
            CNOT_list.append((v0, v1))
        '''form data set for ANN input'''
        map_added = CreateCircuitMapMultiLayer(CNOT_list, num_q_log, num_layer=1)
        #print(map_added.shape)
        #print(data_set[current_lookahead_level].shape)
        data_set[current_lookahead_level] = map_added
    
    '''calculate swap cost via ANN'''
    res = ct.machinelearning.CalSwapCostViaANN(ANN, data_set)
    #res = np.random.randint(19, size=len(level_lookahead))#test only!!!
    '''multiply weights to each level'''
    sum_num_swap = np.sum(weights * res)
    return num_counted_gates, sum_num_swap

def HeuristicCostZhouMLMullayer(ANN, num_q, current_map, DG, executed_vertex,
                                executable_vertex, shortest_length_G,
                                shortest_path_G, num_layer, DiG=None):
    '''
    Calculate heuristic cost for remaining gates and return best path
    this cost is based on the minimial distance in architecture graph between
    two input qubits of each operations
    
    ATTENTION: this function is currently only for bidirectional AG Q20!!!
               for direction AG it need further modification
    
    input:
        ANN -> neural network via keras API
    '''
    mapping = current_map
    num_layer = level_lookahead
    #DG_copy = copy.deepcopy(DG)
    executable_vertex_copy = executable_vertex.copy()
    executed_vertex_copy = executed_vertex.copy()
    current_executable_vertex = executable_vertex_copy
    num_counted_gates = 0
    data_set = np.zeros((1, num_layer, num_q, num_q))
    CNOT_list = []
    for current_lookahead_level in range(num_layer):
        '''lookahead level'''
        #DG_copy.remove_nodes_from(executable_vertex)
        current_executable_vertex = ct.FindExecutableNode(DG,
                                                          executed_vertex_copy,
                                                          current_executable_vertex,
                                                          current_executable_vertex.copy())

        
        num_counted_gates += len(current_executable_vertex)
        for v_DG in current_executable_vertex:
            current_operation = DG.nodes[v_DG]['operation']
            q0 = current_operation.involve_qubits[0]
            q1 = current_operation.involve_qubits[1]
            v0 = mapping.DomToCod(q0)
            v1 = mapping.DomToCod(q1)
            CNOT_list.append((v0, v1))
    '''form data set for ANN input'''
    map_added = CreateCircuitMapMultiLayer(CNOT_list, num_q_log, num_layer=num_layer)
    #print(map_added.shape)
    #print(data_set[current_lookahead_level].shape)
    data_set[0] = map_added
    
    '''calculate swap cost via ANN'''
    sum_num_swap = ct.machinelearning.CalSwapCostViaANN(ANN, data_set)
    #res = np.random.randint(19, size=len(level_lookahead))#test only!!!
    #print(sum_num_swap)
    return num_counted_gates, sum_num_swap[0]

def HeuristicCostZhouMLPolicy(ANN, current_map, DG, executed_vertex,
                              executable_vertex, AG):
    '''
    Calculate heuristic cost for remaining gates and return best path
    this cost is based on the minimial distance in architecture graph between
    two input qubits of each operations
    we will purne the unpromising nodes based on ANN and return their values
    as None
    
    input:
        ANN -> policy neural network via keras API
    '''
    num_layer_ANN = 5
    mapping = current_map
    executable_vertex_copy = executable_vertex.copy()
    executed_vertex_copy = executed_vertex.copy()
    current_executable_vertex = executable_vertex_copy
    num_counted_gates = 0
    num_q_log = len(AG)
    CNOT_list = []
    for current_lookahead_level in range(num_layer_ANN):
        '''lookahead level'''
        if current_lookahead_level == 0:
            current_executable_vertex = current_executable_vertex
        else:
            current_executable_vertex = ct.FindExecutableNode(DG,
                                                              executed_vertex_copy,
                                                              current_executable_vertex,
                                                              current_executable_vertex.copy())

        if len(current_executable_vertex) == 0:
            #return None, np.zeros(len(AG.edges()))
            break
        num_counted_gates += len(current_executable_vertex)
        for v_DG in current_executable_vertex:
            current_operation = DG.nodes[v_DG]['operation']
            q0 = current_operation.involve_qubits[0]
            q1 = current_operation.involve_qubits[1]
            v0 = mapping.LogToPhy(q0)
            v1 = mapping.LogToPhy(q1)
            CNOT_list.append((v0, v1))
    '''form data set for ANN input'''
    #print(CNOT_list)
    map_added = CreateCircuitMapMultiLayer(CNOT_list, num_q_log, 
                                           num_layer=num_layer_ANN)
    #if np.sum(map_added[-1]) == 0:
    #    raise(Exception('error'))
    data_set = np.zeros((1, num_layer_ANN, num_q_log, num_q_log))
    data_set[0] = map_added
    #print(data_set.shape)
    '''calculate swap cost via ANN'''
    swap_pro = ANN.predict(data_set)[0]
    #print(swap_pro)
    return num_counted_gates, swap_pro

def HeuristicCostSAHS(ANN, current_map, DG, executed_vertex,
                      executable_vertex, shortest_length_G,
                      shortest_path_G, level_lookahead, DiG=None):
    '''
    Calculate heuristic cost for remaining gates based on SAHS
    
    ATTENTION: this function is currently only for bidirectional AG Q20!!!
               for direction AG it need further modification
    
    input:
        ANN -> neural network via keras API
    '''
    from machinelearning.data import CreateLabelViaZHOU
    from inputs.operationU import OperationCNOT
    mapping = current_map
    num_layer = len(level_lookahead)
    #DG_copy = copy.deepcopy(DG)
    executable_vertex_copy = executable_vertex.copy()
    executed_vertex_copy = executed_vertex.copy()
    current_executable_vertex = executable_vertex_copy
    num_counted_gates = 0
    data_set = np.zeros((1, num_layer, 20, 20))
    num_q_log = 20
    CNOT_list = []
    for current_lookahead_level in range(num_layer):
        '''lookahead level'''
        #DG_copy.remove_nodes_from(executable_vertex)
        current_executable_vertex = ct.FindExecutableNode(DG,
                                                          executed_vertex_copy,
                                                          current_executable_vertex,
                                                          current_executable_vertex.copy())

        
        num_counted_gates += len(current_executable_vertex)
        for v_DG in current_executable_vertex:
            current_operation = DG.nodes[v_DG]['operation']
            q0 = current_operation.involve_qubits[0]
            q1 = current_operation.involve_qubits[1]
            v0 = mapping.DomToCod(q0)
            v1 = mapping.DomToCod(q1)
            CNOT_list.append(OperationCNOT(v0, v1, []))
    
    '''calculate swap cost via ANN'''
    q_phy = QuantumRegister(20)
    q_log = QuantumRegister(20)
    G = ct.GenerateArchitectureGraph(20, ['IBM QX20'])
    sum_num_swap = CreateLabelViaZHOU(CNOT_list, G, q_phy, q_log,
                                      [shortest_length_G, shortest_length_G],
                                      shortest_path_G)
    return num_counted_gates, sum_num_swap

def HeuristicCostRandom(ANN, current_map, DG, executed_vertex,
                        executable_vertex, shortest_length_G,
                        shortest_path_G, level_lookahead, DiG=None):
    '''
    Calculate heuristic cost for remaining gates based on the Simulation in MCTS
    
    ATTENTION: this function is currently only for bidirectional AG Q20!!!
               for direction AG it need further modification
    
    input:
        ANN -> neural network via keras API
    '''
    from to_python import to_python
    sim_cpp = to_python.SimTest
    mapping = current_map
    num_layer = len(level_lookahead)
    #DG_copy = copy.deepcopy(DG)
    executable_vertex_copy = executable_vertex.copy()
    executed_vertex_copy = executed_vertex.copy()
    current_executable_vertex = executable_vertex_copy
    num_counted_gates = 0
    data_set = np.zeros((1, num_layer, 20, 20))
    num_q_log = 20
    gate0 = []
    gate1 = []
    for current_lookahead_level in range(num_layer):
        '''lookahead level'''
        #DG_copy.remove_nodes_from(executable_vertex)
        current_executable_vertex = ct.FindExecutableNode(DG,
                                                          executed_vertex_copy,
                                                          current_executable_vertex,
                                                          current_executable_vertex.copy())

        
        num_counted_gates += len(current_executable_vertex)
        for v_DG in current_executable_vertex:
            current_operation = DG.nodes[v_DG]['operation']
            q0 = current_operation.involve_qubits[0]
            q1 = current_operation.involve_qubits[1]
            v0 = mapping.DomToCod(q0)
            v1 = mapping.DomToCod(q1)
            gate0.append(v0)
            gate1.append(v1)
    sum_num_swap = sim_cpp(gate0, gate1, list(range(20)), 500)
    return num_counted_gates, sum_num_swap

def HeuristicCostZhou2(ANN, current_map, DG, executed_vertex,
                       executable_vertex, shortest_length_G, 
                       shortest_path_G, level_lookahead, DiG=None):
    '''
    Calculate heuristic cost for remaining gates and return best path
    this cost is based on the minimial distance in architecture graph between two input qubits of each operations
    '''
    worst_num_swap = None
    count_same_worst = 0
    sum_num_swap = 0
    best_num_swap = None
    mapping = current_map
    best_executable_vertex = None
    best_path = None
    if DiG != None: edges = list(DiG.edges)
    #DG_copy = copy.deepcopy(DG)
    executable_vertex_copy = executable_vertex.copy()
    executed_vertex_copy = executed_vertex.copy()
    num_counted_gates = 0
    for current_lookahead_level in range(len(level_lookahead)):
        if current_lookahead_level == 0:
            '''current level'''
            current_executable_vertex = executable_vertex_copy
            weight = 1
        else:
            '''lookahead level'''
            #DG_copy.remove_nodes_from(executable_vertex)
            current_executable_vertex = ct.FindExecutableNode(DG, executed_vertex_copy, current_executable_vertex, current_executable_vertex.copy())
            weight = level_lookahead[current_lookahead_level - 1]
            
        num_counted_gates += len(current_executable_vertex)
        for v_DG in current_executable_vertex:
            flag_4H = 0
            current_operation = DG.nodes[v_DG]['operation']
            q0 = current_operation.involve_qubits[0]
            q1 = current_operation.involve_qubits[1]
            v0 = mapping.DomToCod(q0)
            v1 = mapping.DomToCod(q1)
            current_num_swap = shortest_length_G[v0][v1] - 1
            '''if architecture graph is directed, confirm whether use 4 H gates to change direction'''
            if DiG != None:
                flag_4H = ct.CheckCNOTNeedConvertDirection(v0, v1, shortest_path_G[v0][v1], edges)
            current_num_swap += flag_4H * 4/7
            '''renew number of all swaps'''
            current_num_swap = current_num_swap * weight#multiply the weight for cost of gates in different levels
            sum_num_swap = sum_num_swap + current_num_swap
            '''renew swap number of worst operation'''
            if worst_num_swap == None:
                worst_num_swap = current_num_swap
                count_same_worst = 0
            else:
                if current_num_swap > worst_num_swap:
                    worst_num_swap = current_num_swap
                    count_same_worst = 0
                else:
                    if current_num_swap == worst_num_swap:
                        count_same_worst += 1
            '''renew swap number of best operation'''
            if best_num_swap == None:
                best_num_swap = current_num_swap
                best_path = shortest_path_G[v0][v1]
                best_executable_vertex = v_DG
            else:
                if current_num_swap < best_num_swap:
                    best_num_swap = current_num_swap
                    best_path = shortest_path_G[v0][v1]
                    best_executable_vertex = v_DG
        
    return num_counted_gates, sum_num_swap
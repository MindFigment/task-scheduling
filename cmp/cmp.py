import sys
import click
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from operator import add
import warnings

plt.style.use('seaborn-poster')

class CMP(object):

    def __init__(self, graph_file):
        self.nodes_dict = defaultdict(dict)
        self.graph = self._create_graph_form_file(graph_file)
        self._is_dag(self.graph)
        self.topological_sort = self._get_topological_order()
        self._compute_earliest_execution()
        self.critical_nodes, self.critical_edges = self._compute_latest_execution()
        self.critical_path = self._find_critical_path(self.critical_edges)
        print("CRITICAL NODES: ", self.critical_nodes)
        print("CRITICAL EDGES: ", self.critical_edges)
        print("CRITICAL PATH: ", self.critical_path)


    def _create_graph_form_file(self, graph_file):
        G = nx.DiGraph()
        process_weights = True
        with open(graph_file) as f:
            print("READING GRAPH FROM FILE")
            for line in f:
                if line == '\n':
                    process_weights = False
                elif process_weights:
                    node, weight = line.strip().split(' ')
                    print((node, weight))
                    G.add_node(node)
                    self.nodes_dict[node]['weight'] = int(weight)
                else:
                    node1, node2 = line.strip().replace(' ', '').split('->')
                    G.add_edge(node1, node2)
        return G


    def _is_dag(self, graph):
        is_dag = nx.is_directed_acyclic_graph(graph)
        if is_dag:
            print("All good, graph is a DAG")
        else:
            raise ValueError("Graph is not a DAG!")


    def _compute_earliest_execution(self):
        for node in self.topological_sort:   
            predecessors = self._get_node_predecessors(node)
            earliest_exec_times = list(map(add, self._get_from_dict(predecessors, 'ee', default=[0]), self._get_from_dict(predecessors, 'weight',  default=[0])))
            self.nodes_dict[node]['ee'] = max(earliest_exec_times)
            

    def _compute_latest_execution(self):
        # find the longest path and change latest execution time in all end nodes for it
        end_nodes = self._get_end_nodes()
        critical_end_time = max(list(map(add, self._get_from_dict(end_nodes, 'ee'), self._get_from_dict(end_nodes, 'weight'))))
        for node in end_nodes:
            self.nodes_dict[node]['le'] = critical_end_time

        critical_nodes = []
        critical_edges = []

        # my latest execution = successor earliest execution - my execution time
        # if no succesor latest execution = critical end execution - my execution time
        for node in reversed(self.topological_sort):   
            successors = self._get_node_successors(node)
            successors_earliest_exec_times = self._get_from_dict(successors, 'ee', default=[critical_end_time])
            self.nodes_dict[node]['le'] = min(successors_earliest_exec_times) - self.nodes_dict[node]['weight']
            if self.nodes_dict[node]['le'] == self.nodes_dict[node]['ee']:
                critical_nodes.append(node)
                critical_edges.extend(self._create_critical_edges(node, self._find_critical_successors(node, successors)))

        return critical_nodes, critical_edges


    def _find_critical_path(self, critical_edges):
        end_nodes = self._get_end_nodes()
        for j, (snode, enode) in enumerate(critical_edges):
            if enode in end_nodes:
                start_node = snode
                end_node = enode
                i = j
                break
        critical_path = [(start_node, end_node)]
        while self._find_next_edge(start_node, critical_edges[i+1:]) != None:
            offset, next_edge = self._find_next_edge(start_node, critical_edges[i+1:])
            i += offset
            critical_path.append(next_edge)
            start_node = next_edge[0]
        return critical_path


    def _find_critical_successors(self, critical_node, successors):
        critical_succesors = []
        for successor in successors:
            if self.nodes_dict[successor]['le'] == (self.nodes_dict[critical_node]['le'] + self.nodes_dict[critical_node]['weight']):
                critical_succesors.append(successor)
        return critical_succesors


    def _create_critical_edges(self, critical_node, critical_succesors):
        return [(critical_node, cs) for cs in critical_succesors]


    def _find_next_edge(self, start_node, remaining_edges):
        for i, (start, end) in enumerate(remaining_edges):
            if end == start_node:
                return i, (start, end)
        return None


    def _get_node_predecessors(self, node):
        return list(self.graph.predecessors(node))


    def _get_node_successors(self, node):
        return list(self.graph.successors(node))

    
    def _get_start_nodes(self):
        output = []
        for node in self.graph.nodes():
            if len(self._get_node_predecessors(node)) == 0:
                output.append(node)
        return output


    def _get_end_nodes(self):
        output = []
        for node in self.graph.nodes():
            if len(self._get_node_successors(node)) == 0:
                output.append(node)
        return output


    def _get_topological_order(self):
        return list(nx.topological_sort(self.graph))


    def _get_from_dict(self, keys, field, default=None):
        if not keys:
            return default
        else:
            return [self.nodes_dict[key][field] if field in self.nodes_dict[key] else 0 for key in keys]


    def create_timetable(self):
        begin_task = []
        end_task = []
        machines = []
        free_from = []
        task_names = []
        critical_path_nodes = set(self._flatten_list(self.critical_path))
        max_path = 0
        for node in critical_path_nodes:
            begin = self.nodes_dict[node]['ee']
            end = begin + self.nodes_dict[node]['weight']
            begin_task.append(begin)
            end_task.append(end)
            machines.append(0)
            task_names.append(node)
            max_path += self.nodes_dict[node]['weight']

        free_from.append(max_path)
        free_from.append(0)
        num_machines = 2
        task_added = False
        rest_nodes = [node for node in self.topological_sort if node not in critical_path_nodes]
        for node in rest_nodes:
            for machine in range(num_machines):
                if free_from[machine] <= self.nodes_dict[node]['le']:
                    begin_task.append(max(self.nodes_dict[node]['ee'], free_from[machine]))
                    end_task.append(begin_task[-1] + self.nodes_dict[node]['weight'])
                    machines.append(machine)
                    free_from[machine] = max(self.nodes_dict[node]['ee'], free_from[machine]) + self.nodes_dict[node]['weight']
                    task_names.append(node)
                    task_added = True
                    break
            if not task_added:        
                begin_task.append(self.nodes_dict[node]['ee'])
                end_task.append(begin_task[-1] + self.nodes_dict[node]['weight'])
                machines.append(num_machines)
                free_from.append(end_task[-1])
                task_names.append(node)
                num_machines += 1
            task_added = False

        return begin_task, end_task, machines, task_names
        
    def _flatten_list(self, flatten_me):
        return [item for sublist in flatten_me for item in sublist]

    
def draw_graph(cmp):
    warnings.filterwarnings('ignore', category=FutureWarning)
    _ = plt.figure()
    graph = cmp.graph
    nodes_dict = cmp.nodes_dict
    critical_nodes = cmp.critical_nodes
    critical_edges = cmp.critical_edges
    critical_path = cmp.critical_path
    critical_edges = set(critical_edges).difference(set(critical_path))
    node_color = ['firebrick' if n in critical_nodes else 'cornflowerblue' for n in graph.nodes()]
    edge_color = ['firebrick' if n in critical_path else 'olive' if n in critical_edges else 'black' for n in graph.edges()]
    # # node positions
    pos = nx.planar_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=3000, node_color=node_color)
    nx.draw_networkx_edges(graph, pos, node_size=3000, arrowsize=20, edge_color=edge_color, alpha=0.7)
    node_labels = dict((n, str(n + '\n(' + str(d['ee']) + ', ' + str(d['ee'] + d['weight']) + ')' + '\n(' + str(d['le']) + ', ' + str(d['le'] + d['weight']) + ')')) for n, d in nodes_dict.items())
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8, font_family='sans-serif')
    plt.axis('off')

    
def draw_timetable(begin_task, end_task, indices, names):
    _ = plt.figure()
    begin_task = np.array(begin_task)
    end_task =   np.array(end_task)
    indices = np.array(indices)
    machines = ["Machine {}".format(i+1) for i in range(len(set(indices)))]

    barh = plt.barh(indices, end_task-begin_task, left=begin_task, edgecolor='black')

    for rect, label in zip(barh, names):
        height = rect.get_height()  
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()
        plt.text(x + width / 2, y + height / 2, label, ha='center', va='bottom', color='black')

    plt.yticks(range(len(machines)), machines)


@click.command()
@click.argument("graph_file")
def main(graph_file):
    cmp = CMP(graph_file)
    draw_graph(cmp)
    e, b, i, n = cmp.create_timetable()
    draw_timetable(e, b, i, n)
    plt.show()

if __name__ == "__main__":
    main()

import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import click 
import sys
import warnings

plt.style.use('seaborn-poster')


class CoffmanGrahan(object):

    def __init__(self, graph_file):
        self.nodes_dict = defaultdict(dict)  # defaultdict(lambda: defaultdict(list))
        self.graph = self._create_graph_form_file(graph_file)
        self._is_dag()
        self.task_order = None


    def _create_graph_form_file(self, graph_file):
        G = nx.DiGraph()
        with open(graph_file) as f:
            print("READING GRAPH FROM FILE")
            for line in f:
                print(line.replace(' ', '')[:-1])
                node1, node2 = line.strip().replace(' ', '').split('->')
                G.add_edge(node1, node2)
        return G


    def _is_dag(self):
        is_dag = nx.is_directed_acyclic_graph(self.graph)
        if is_dag:
            print("All good, graph is a DAG")
        else:
            raise ValueError("Graph is not a DAG!")


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

    
    def _get_succesors_sorted_labels(self, node):
        output = list(map(lambda n: self.nodes_dict[n]['label'], self._get_node_successors(node)))
        # print(sorted(output, reverse=True))
        return sorted(output, reverse=True)


    def _check_if_all_predecessors_successors_labeled(self, node):
        output = []
        predecessors = self._get_node_predecessors(node)
        for pred in predecessors:
            # check_if_all_labeled(nodes):
            # print("NODE SUCCESORS", self._get_node_successors(pred))
            if all(['label' in self.nodes_dict[node] for node in self._get_node_successors(pred)]):
                output.append(pred)
        return output


    def run(self):
        reversed_task_order = []
        A = self._get_end_nodes()
        for i in range(len(self.graph.nodes())):
            d = {}
            for z in A:
                s_list = self._get_succesors_sorted_labels(z)
                # print(z, "S_LIST", s_list)
                d[z] = s_list
                self.nodes_dict[z]['s_list'] = s_list
                
            sorted_nodes = sorted(d, key=lambda x: [len(d[x])] + d[x])
            print("SORTED NODES", sorted_nodes)    
            for k, v in d.items():
                print(k, [len(v)] + v)

            labaled_node = sorted_nodes.pop(0)
            self.nodes_dict[labaled_node]['label'] = i + 1
            reversed_task_order.append(labaled_node)
            A.remove(labaled_node)
            ext = self._check_if_all_predecessors_successors_labeled(labaled_node)
            A.extend(ext)

        # Reversing task order
        self.task_order = reversed_task_order[::-1]
        print("TASK ORDER:", self.task_order)


    def create_timetable(self, num_machines):
        begin_task = []
        end_task = []
        machine = []
        task_names = []
        task_order = list(self.task_order)
        current_machine = 1
        current_time = 0
        current_task = task_order[0]
        current_task_index = 0
        task_end_time = {}
        while task_order != []:
            print("CURRENT TASK", current_task)
            if all(task not in task_order and task_end_time[task] <= current_time for task in self._get_node_predecessors(current_task)):
                begin_task.append(current_time)
                end_task.append(current_time + 1)
                task_end_time[current_task] = current_time + 1
                machine.append(current_machine - 1)
                task_names.append(current_task)
                print("Remove", current_task, "from", task_order)
                task_order.remove(current_task)
    
                if task_order != []:
                    if current_task_index < len(task_order) and current_machine < num_machines:
                        # No need to increment currint index after removing task from task_order list
                        current_task = task_order[current_task_index]
                    else:
                        current_task = task_order[0]
                        current_task_index = 0
                        current_time += 1

                current_machine = (current_machine % num_machines) + 1

            elif current_task_index + 1 < len(task_order):
                current_task = task_order[current_task_index + 1]
                current_task_index += 1

            else:
                current_time += 1
                current_task_index = 0
                current_task = task_order[0]
                current_machine = 1

            # print(">.................", current_task, current_task_index)

        return begin_task, end_task, machine, task_names



def draw_graph(cof_gra): 
    warnings.filterwarnings('ignore', category=FutureWarning)
    _ = plt.figure()
    graph = cof_gra.graph
    nodes_dict = cof_gra.nodes_dict
    # pos = nx.circular_layout(graph)  
    pos = nx.kamada_kawai_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=1000)
    nx.draw_networkx_edges(graph, pos, node_size=1000, arrowsize=20, alpha=0.7)
    node_labels = dict((n, str(n + '\n' + str(d['label']) + '\n' + str(d['s_list']))) for n, d in nodes_dict.items())
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8, font_family='sans-serif')
    plt.axis('off')


def draw_timetable(cof_gra, machines_num):
    _ = plt.figure()
    begin_task, end_task, indices, names = cof_gra.create_timetable(machines_num)
    begin_task = np.array(begin_task)
    end_task =   np.array(end_task)
    indices = np.array(indices)
    machines = ["Machine {}".format(i) for i in range(len(set(indices)))]

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
@click.argument("machines_num", type=click.INT)
def main(graph_file, machines_num):
    cof_gra = CoffmanGrahan(graph_file)
    cof_gra.run()
    draw_graph(cof_gra)
    draw_timetable(cof_gra, machines_num)
    plt.show()



if __name__ == "__main__":
    main()
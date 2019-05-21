import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import click 
import sys
import warnings
import matplotlib.cm as cm

plt.style.use('seaborn-poster')


class LiuModified(object):

    def __init__(self, graph_file):
        self.nodes_dict = defaultdict(dict)  # defaultdict(lambda: defaultdict(list))
        self.graph = self._create_graph_form_file(graph_file)
        self._is_dag()
        self.task_order = None
        self._compute_d_modified()


    def _create_graph_form_file(self, graph_file):
        G = nx.DiGraph()
        process_weights = True
        index = 0
        with open(graph_file) as f:
            print("READING GRAPH FROM FILE")
            for line in f:
                if line == '\n':
                    process_weights = False
                elif process_weights:
                    node, p, d, r = line.strip().split(' ')
                    print((node, p, d, r))
                    G.add_node(node)
                    self.nodes_dict[node]['p'] = int(p)
                    self.nodes_dict[node]['d'] = int(d)
                    self.nodes_dict[node]['r'] = int(r)
                    self.nodes_dict[node]['i'] = index
                    index += 1
                else:
                    node1, node2 = line.strip().replace(' ', '').split('->')
                    G.add_edge(node1, node2)
        return G


    def _compute_d_modified(self):
        for task in self.nodes_dict.keys():
            self.nodes_dict[task]['d*'] = self._compute_d_modified_for_task(task)


    def _compute_d_modified_for_task(self, task):
        tasks = [task]
        tasks.extend(self._get_all_dependent_tasks(task))
        d_min = self._compute_min_d(tasks)
        return d_min


    def _get_all_dependent_tasks(self, task):
        output = [task]
        successors = self._get_node_successors(task)
        for succ in successors:
            tasks = self._get_all_dependent_tasks(succ)
            for task in tasks:
                if task not in output:
                    output.append(task)
        return output


    def _compute_min_d(self, tasks):
        output = min([self.nodes_dict[task]['d'] for task in tasks])
        return output


    def _is_dag(self):
        is_dag = nx.is_directed_acyclic_graph(self.graph)
        if is_dag:
            print("All good, graph is a DAG")
        else:
            raise ValueError("Graph is not a DAG!")


    def _get_node_successors(self, node):
        return list(self.graph.successors(node))

    def _get_node_predecessors(self, node):
        return list(self.graph.predecessors(node))


    def run(self):
        self._compute_d_modified()


    def create_timetable(self):
        begin_task = []
        end_task = []
        task_names = []
        tasks_ready_to_execute = []
        tasks_not_in_the_system = []
        tasks_in_the_system_waiting_for_predecessors = []
        tasks_executed = []
        task_num = 0
        
        time_left = {}
        for task, data_dict in self.nodes_dict.items():
            time_left[task] = data_dict['p']
            task_num += 1
            tasks_not_in_the_system.append(task)

        tasks_to_remove = []
        current_time = 0
        while len(tasks_executed) != task_num:
            for task in tasks_not_in_the_system:
                if self.nodes_dict[task]['r'] <= current_time:
                    predecessors = self._get_node_predecessors(task)
                    can_add = all([pred in tasks_executed for pred in predecessors])
                    if can_add:
                        tasks_ready_to_execute.append(task)
                        tasks_ready_to_execute = sorted(tasks_ready_to_execute, key=lambda n:self.nodes_dict[n]['d*'], reverse=False)
                    else:
                        tasks_in_the_system_waiting_for_predecessors.append(task)
                    tasks_to_remove.append(task)

            for task in tasks_to_remove:
                tasks_not_in_the_system.remove(task)
            tasks_to_remove = []

            if len(tasks_ready_to_execute) >= 1:

                next_task_to_execute = tasks_ready_to_execute[0]
                begin_task.append(current_time)
                end_task.append(current_time + 1)
                task_names.append(next_task_to_execute)
                time_left[next_task_to_execute] -= 1
                if time_left[next_task_to_execute] == 0:
                    tasks_ready_to_execute.remove(next_task_to_execute)
                    tasks_executed.append(next_task_to_execute)
                    self.nodes_dict[next_task_to_execute]['L'] = current_time + 1 - self.nodes_dict[next_task_to_execute]['d']
                    for task in tasks_in_the_system_waiting_for_predecessors:
                        predecessors = self._get_node_predecessors(task)
                        can_add = all([pred in tasks_executed for pred in predecessors])
                        if can_add:
                            tasks_ready_to_execute.append(task)
                            tasks_ready_to_execute = sorted(tasks_ready_to_execute, key=lambda n:self.nodes_dict[n]['d*'])
                            tasks_in_the_system_waiting_for_predecessors.remove(task)

                # print('time left:', [t for t in time_left.values()])
                # print('tasks not in the system:', tasks_not_in_the_system)
                # print('tasks in the system waiting for succesors:', tasks_in_the_system_waiting_for_predecessors)
                # print('tasks ready to execute:', tasks_ready_to_execute)
                # print('tasks executed:',tasks_executed)

            current_time += 1


        return begin_task, end_task, task_names, current_time + 1



def draw_graph(liu_mod): 
    warnings.filterwarnings('ignore', category=FutureWarning)
    _ = plt.figure()
    graph = liu_mod.graph
    nodes_dict = liu_mod.nodes_dict
    # pos = nx.kamada_kawai_layout(graph) 
    pos = nx.circular_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=2000, node_color='olive')
    nx.draw_networkx_edges(graph, pos, node_size=2000, arrowsize=20, alpha=0.7)
    node_labels = dict((n, str(n + '\n' + 'd:' + str(d['d'])+ ' d*:' + str(d['d*']) + '\n' + 'r:' + str(d['r']) + ' p:' + str(d['p']))) for n, d in nodes_dict.items())
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8, font_family='sans-serif')
    plt.axis('off')


def draw_timetable(liu_mod):
    _ = plt.figure()
    begin_task, end_task, names, time_sum = liu_mod.create_timetable()
    begin_task = np.array(begin_task)
    end_task =   np.array(end_task)
    indices = np.zeros_like(names)
    machines = ["Machine 1"]

    colors = cm.rainbow(np.linspace(0, 1, len((list(set(names))))))

    task_colors = [ colors[liu_mod.nodes_dict[task]['i']] for task in names ]

    barh = plt.barh(indices, end_task-begin_task, left=begin_task, edgecolor='black', color=task_colors)

    for rect, label in zip(barh, names):
        height = rect.get_height()  
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()
        plt.text(x + width / 2, y + height / 2, label, ha='center', va='bottom', color='black')

    plt.yticks(range(len(machines)), machines)
    plt.xticks(range(time_sum))


@click.command()
@click.argument("graph_file")
def main(graph_file):
    liu_mod = LiuModified(graph_file)
    liu_mod.run()
    draw_graph(liu_mod)
    draw_timetable(liu_mod)
    L = [v['L'] for v in liu_mod.nodes_dict.values()]
    print(L)
    print('Lmax:', max(L))
    plt.show()



if __name__ == "__main__":
    main()
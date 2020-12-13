import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import click 
import sys
import warnings
import matplotlib.cm as cm

plt.style.use('seaborn-poster')


class JohnsonModified(object):

    def __init__(self, tasks_file):
        self.tasks = self._get_tasks_form_file(tasks_file)
        self._is_dominated()


    def _get_tasks_form_file(self, tasks_file):
        tasks = defaultdict(list)
        with open(tasks_file) as f:
            print("READING TASKS FROM FILE")
            machine_data = f.readline().split(' ')
            machine = machine_data[0]
            tasks[machine] =  [int(task) for task in machine_data[1:]]
            length = len(machine_data)
            print(machine, tasks[machine])
            machines_num = 1
            for line in f:
                machine_data = line.split(' ')
                if len(machine_data) != length:
                    raise ValueError("Wrong number of tasks!")
                machine = machine_data[0]
                tasks[machine] = [int(task) for task in machine_data[1:]]
                machines_num += 1
                print(machine, tasks[machine])

            if machines_num != 3:
                raise ValueError("Wrong number of machines", machines_num)

        return tasks


    def _is_dominated(self):
        m1 = self.tasks['M1']
        m2 = self.tasks['M2']
        m3 = self.tasks['M3']
        min1 = min(m1)
        max2 = max(m2)
        min3 = min(m3)
        if all([t1 >= t2 for t1, t2 in zip(m1, m2)]) and min1 >= max2:
            print("M1 dominates M2")
        elif all([t3 >= t2 for t3, t2 in zip(m3, m2)]) and min3 >= max2:
            
            print("M3 dominates M2")
        else:
            raise ValueError("M2 is not dominated!")


    def compute_modified_time(self):
        m1 = self.tasks['M1']
        m2 = self.tasks['M2']
        m3 = self.tasks['M3']
        t1 = [p1 + p2 for p1, p2 in zip(m1, m2)]
        t2 = [p2 + p3 for p2, p3 in zip(m2, m3)]
        self.tasks['T1'] = t1
        self.tasks['T2'] = t2
        n1 = []
        n2 = []
        for i, [x1, x2] in enumerate(zip(t1, t2)):
            if x1 < x2:
                n1.append([i, x1])
            else:
                n2.append([i, x2])

        print("M1", m1)
        print("M2", m2)
        print("M3", m3)

        print("T1", t1)
        print("T2", t2)

        n1 = sorted(n1, key=lambda n:n[1], reverse=False)
        n2 = sorted(n2, key=lambda n:n[1], reverse=True)

        print("N1", n1)
        print("N2", n2)

        self.tasks['N1'] = n1
        self.tasks['N2'] = n2


    def run(self):
        self.compute_modified_time()


    def create_timetable(self):
        task_order = self.tasks['N1'].copy()
        task_order.extend(self.tasks['N2'])
        
        print("TASK ORDER", task_order)
        print(self.tasks['N1'])
        print(self.tasks['N2'])

        m1 = self.tasks['M1']
        m2 = self.tasks['M2']
        m3 = self.tasks['M3']

        begin_task = []
        end_task = []
        machine = []
        names = []
        task_num = []
        t1 = 0
        t2 = 0
        t3 = 0
        
        for task, _ in task_order:
            # machine 1
            begin_task.append(t1)
            end_task.append(t1 + m1[task])
            t1 += m1[task]
            # machine 2
            t2 = max(t1, t2)
            begin_task.append(t2)
            end_task.append(t2 + m2[task])
            t2 += m2[task]
            # machine 3
            t3 = max(t2, t3)
            begin_task.append(t3)
            end_task.append(t3 + m3[task])
            t3 += m3[task]
            # append machines
            machine.extend([0, 1, 2])
            task_name = 'Z' + str(task + 1)
            names.extend([task_name, task_name, task_name])
            task_num.extend([task, task, task])

        return begin_task, end_task, machine, names, task_num, max(t1, t2, t3)

            
        


def draw_timetable(johnson_mod):
    _ = plt.figure()
    begin_task, end_task, machines, names, task_num, time_sum = johnson_mod.create_timetable()
    begin_task = np.array(begin_task)
    end_task =   np.array(end_task)
    indices = np.array(machines)

    colors = cm.rainbow(np.linspace(0, 1, len(list(set(task_num)))))

    task_colors = [colors[i] for i in task_num]

    barh = plt.barh(indices, end_task-begin_task, left=begin_task, edgecolor='black', color=task_colors)

    for rect, label in zip(barh, names):
        height = rect.get_height()  
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()
        plt.text(x + width / 2, y + height / 2, label, ha='center', va='bottom', color='black')

    plt.yticks(range(3), ['M1', 'M2', 'M3'])
    plt.xticks(range(time_sum + 1))


@click.command()
@click.argument("tasks_file")
def main(tasks_file):
    johnson_mod = JohnsonModified(tasks_file)
    johnson_mod.run()
    draw_timetable(johnson_mod)
    plt.show()



if __name__ == "__main__":
    main()
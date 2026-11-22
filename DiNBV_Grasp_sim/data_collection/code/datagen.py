"""
    Batch-generate data
"""

import os
import numpy as np
import multiprocessing as mp
from subprocess import call
from utils import printout
import time


class DataGen(object):

    def __init__(self, num_processes, flog=None):
        self.num_processes = num_processes
        self.flog = flog
        
        self.todos = []
        self.processes = []
        self.is_running = False
        self.Q = mp.Queue()

    def __len__(self):
        return len(self.todos)

    def add_one_collect_job(self, data_dir, shape_id, category, cnt_id, primact_type, trial_id, camera_distance=None, use_wrist_camera=False, save_approach_trajectory=False, distance_step=0.1, save_rich_features=False):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot add a new job while DataGen is running!')
            exit(1)

        todo = ('COLLECT', shape_id, category, cnt_id, primact_type, data_dir, trial_id, np.random.randint(10000000), camera_distance, use_wrist_camera, save_approach_trajectory, distance_step, save_rich_features)
        self.todos.append(todo)
    
    def add_one_recollect_job(self, src_data_dir, dir1, dir2, recollect_record_name, tar_data_dir, x, y):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot add a new job while DataGen is running!')
            exit(1)

        todo = ('RECOLLECT', src_data_dir, recollect_record_name, tar_data_dir, np.random.randint(10000000), x, y, dir1, dir2)
        self.todos.append(todo)
    
    def add_one_checkcollect_job(self, src_data_dir, dir1, dir2, recollect_record_name, tar_data_dir, x, y):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot add a new job while DataGen is running!')
            exit(1)

        todo = ('CHECKCOLLECT', src_data_dir, recollect_record_name, tar_data_dir, np.random.randint(10000000), x, y, dir1, dir2)
        self.todos.append(todo)
    
    @staticmethod
    def job_func(pid, todos, Q):
        """
        Worker process entry point.

        For each COLLECT job we always call `collect_data.py` (the latest variant).
        Legacy rich-feature scripts have been removed for GitHub release.
        """
        succ_todos = []
        for todo in todos:
            if todo[0] == 'COLLECT':
                # Run without GUI
                script_name = 'collect_data.py'

                cmd = 'xvfb-run -a python %s %s %s %d %s --out_dir %s --trial_id %d --random_seed %d --no_gui' \
                        % (script_name, todo[1], todo[2], todo[3], todo[4], todo[5], todo[6], todo[7])
                # Add camera_distance parameter if specified
                if len(todo) > 8 and todo[8] is not None:
                    cmd += ' --camera_distance %s' % todo[8]
                # Add use_wrist_camera flag if specified
                if len(todo) > 9 and todo[9]:
                    cmd += ' --use_wrist_camera'
                # Add save_approach_trajectory flag if specified
                if len(todo) > 10 and todo[10]:
                    cmd += ' --save_approach_trajectory'
                # Add distance_step parameter if specified
                if len(todo) > 11 and todo[11] != 0.1:
                    cmd += ' --distance_step %s' % todo[11]
                # Add save_rich_features flag if specified
                if len(todo) > 12 and todo[12]:
                    cmd += ' --save_rich_features'

                folder_name = todo[5]
                job_name = '%s_%s_%d_%s_%s' % (todo[1], todo[2], todo[3], todo[4], todo[6])
                ret = call(cmd, shell=True)
                if ret == 0:
                    succ_todos.append(os.path.join(folder_name, job_name))
                if ret == 2:
                    succ_todos.append(None)
        Q.put(succ_todos)

    def start_all(self):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot start all while DataGen is running!')
            exit(1)

        total_todos = len(self)
        num_todos_per_process = int(np.ceil(total_todos / self.num_processes))
        np.random.shuffle(self.todos)
        for i in range(self.num_processes):
            todos = self.todos[i*num_todos_per_process: min(total_todos, (i+1)*num_todos_per_process)]
            p = mp.Process(target=self.job_func, args=(i, todos, self.Q))
            p.start()
            self.processes.append(p)
        
        self.is_running = True

    def join_all(self):
        if not self.is_running:
            printout(self.flog, 'ERROR: cannot join all while DataGen is idle!')
            exit(1)

        ret = []
        for p in self.processes:
            ret += self.Q.get()

        for p in self.processes:
            p.join()

        self.todos = []
        self.processes = []
        self.Q = mp.Queue()
        self.is_running=False
        return ret



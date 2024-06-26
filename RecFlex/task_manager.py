# Copyright 2024 The RecFlex Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import queue
import threading
import multiprocessing as mp

from typing import List, Any


class TaskManager:
    def __init__(self, resources: List[Any], max_threads: int = 256):
        self.resource_queue = queue.Queue(len(resources))
        for resource in resources:
            self.resource_queue.put(resource)
        self.task_threads = []
        self.max_threads = max_threads

        assert mp.get_start_method() == 'spawn', "Plase add multiprocessing.set_start_method('spawn') in the start of your program!"

    def _exec(self, target, args, ident_key: str = None):
        ident = self.resource_queue.get()
        try:
            if ident_key:
                p = mp.Process(target=target, args=args, kwargs={ ident_key: ident })
            else:
                p = mp.Process(target=target, args=args)
            p.start()
            p.join()
        finally:
            self.resource_queue.put(ident)

    def add_task(self, target, args, ident_key: str = None):
        thread = threading.Thread(target=self._exec, args=(target, args, ident_key))
        self.task_threads.append(thread)
        thread.start()

        if len(self.task_threads) >= self.max_threads:
            print("Too many tasks have been added! Forced to wait for previous threads finishing.")
            self.join()

    def join(self):
        for thread in self.task_threads:
            thread.join()
        self.task_threads = []

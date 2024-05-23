import multiprocessing
import threading
import collections

import queue
import math
import os
import torch
import numpy as np
from torch._six import string_classes
from torch._utils import ExceptionWrapper
from torch.utils.data import SequentialSampler, BatchSampler

MP_STATUS_CHECK_INTERVAL = 5.0

def has_same_shape(tensors):
    """===
    """
    shapes = [np.array(t.shape) for t in tensors]
    for shape in shapes:
        if len(shape) != len(shapes[0]):
            return False
        if (shape != shapes[0]).any():
            return False
    return True

def default_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    for e in batch:
        if type(e) != elem_type:
            return batch
    if isinstance(elem, torch.Tensor):
        if has_same_shape(batch):
            return torch.stack(batch, 0)
        return batch
    elif isinstance(elem, dict):
        keys = set()
        for e in batch:
            keys = keys | set(e.keys())
        batch_dict = dict()
        for key in keys:
            batch_temp = [e.get(key, None) for e in batch]
            batch_dict[key] = default_collate(batch_temp)
        batch = batch_dict
    else:
        return batch
    return batch



def pin_memory(data, device=None):
    if isinstance(data, torch.Tensor):
        return data.pin_memory(device)
    elif isinstance(data, string_classes):
        return data
    elif isinstance(data, collections.abc.Mapping):
        try:
            return type(data)({k: pin_memory(sample, device) for k, sample in data.items()})  # type: ignore[call-arg]
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {k: pin_memory(sample, device) for k, sample in data.items()}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return type(data)(*(pin_memory(sample, device) for sample in data))
    elif isinstance(data, tuple):
        return [pin_memory(sample, device) for sample in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence):
        try:
            return type(data)([pin_memory(sample, device) for sample in data])  # type: ignore[call-arg]
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [pin_memory(sample, device) for sample in data]
    elif hasattr(data, "pin_memory"):
        return data.pin_memory()
    else:
        return data


def batch2device(batch, device, data_trans=None):
    """
    convert data device
    """
    def to_device_recurssive(data, device):
        """no_doc
        """
        if isinstance(data, torch.Tensor):
            if data.device == device:
                return data
            return data.to(device, non_blocking=True)
        elif isinstance(data, list):
            return [to_device_recurssive(d, device) for d in data]
        elif isinstance(data, dict):
            for key in data:
                data[key] = to_device_recurssive(data[key], device)
            return data
        else:
            return data
    batch = to_device_recurssive(batch, device)
    if data_trans is not None:
        batch = data_trans(batch)
    return batch


def _preload_fun_loop(preproc_fun, in_queue, out_queue, done_event, device=torch.device('cpu')):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)
    if 'cuda' in device.type:
        torch.cuda.set_device(device)

    def do_one_step():
        if out_queue.full():
            return
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            return
        data, idx, bs = r
        data = preproc_fun(data, device)
        while not done_event.is_set():
            try:
                out_queue.put((data, idx, bs), timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except:
                pass

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    while not done_event.is_set():
        # Make sure that we don't preserve any object from one iteration
        # to the next
        do_one_step()


class ManagerWatchdog(object):  # type: ignore[no-redef]
    def __init__(self):
        self.manager_pid = os.getppid()
        self.manager_dead = False

    def is_alive(self):
        if not self.manager_dead:
            self.manager_dead = os.getppid() != self.manager_pid
        return not self.manager_dead

class _SingleProcessDataLoaderIter:
    def __init__(self, loader) -> None:
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.collate_fn = loader.collate_fn

    def __iter__(self):
        return self
    
    def reset(self):
        self.index = 0
    
    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        batch_size = min(len(self.dataset) - self.index, self.batch_size)
        batch = self.collate_fn([self.dataset[self.index + i] for i in range(batch_size)])
        self.index += batch_size
        return batch


class _MultiProcessingDataLoaderIter:
    def __init__(self, loader) -> None:
        self.loader = loader
        self.dataset = loader.dataset
        self.num_workers = loader.num_workers
        self.prefetch_factor = loader.prefetch_factor
        self.batch_sampler = loader.batch_sampler
        self.persistent_workers = loader.persistent_workers
        self.preload = loader.preload
        self.preproc_fun = loader.preproc_fun
        self._worker_result_queue = multiprocessing.JoinableQueue()
        self._workers_done_event = multiprocessing.Event()
        self.workers = []
        self.index_queues = []
        for i in range(self.num_workers):
            index_queue = multiprocessing.Queue(self.prefetch_factor)
            worker = multiprocessing.Process(target=self._worker_loop, args=(i, self.dataset, index_queue, 
                    self._worker_result_queue, loader.collate_fn, self._workers_done_event, self.persistent_workers,))
            worker.start()
            self.workers.append(worker)
            self.index_queues.append(index_queue)
        
        if self.preload:
            self._pre_load_thread_done_event = threading.Event()
            self._data_queue = queue.Queue(self.num_workers)  # type: ignore[var-annotated]
            pre_load_thread = threading.Thread(target=_preload_fun_loop, args=(self.preproc_fun, 
                    self._worker_result_queue, self._data_queue, self._workers_done_event, loader.preload_device))
            pre_load_thread.daemon = True
            pre_load_thread.start()
            self._pre_load_thread = pre_load_thread
            self.data_queue = self._data_queue
        else:
            self.data_queue =self._worker_result_queue

        self.reset()

    def __iter__(self):
        return self
    
    def reset(self):
        self.index = 0
        for que in self.index_queues:
            while not que.empty():
                que.get()
        while not self.data_queue.empty():
            self.data_queue.get()
        
        self.idle_workers = queue.Queue()
        for idx in [i for i in range(self.num_workers) for _ in range(self.prefetch_factor)][::-1]:
            self.idle_workers.put(idx)
        self._sampler_iter = iter(self.batch_sampler)
        for i in range(self.num_workers * self.prefetch_factor):
            ret = self._try_put_index()
            if not ret:
                break

    def shutdown(self):
        self._workers_done_event.set()
        if self.preload:
            self._pre_load_thread_done_event.set()
        # for worker in self.workers:
        #     worker.join()

    def _try_put_index(self):
        try:
            indexs = next(self._sampler_iter)
            sel_worker = self.idle_workers.get()
            self.index_queues[sel_worker].put(indexs)
            return True
        except StopIteration:
            return False
        
    def _try_get_data(self):
        if self.index >= len(self.dataset):
            if not self.persistent_workers:
                self.shutdown()
            raise StopIteration
        batch = self.data_queue.get()
        # put next index
        self.idle_workers.put(batch[1])
        self._try_put_index()
        self.index += batch[2]
        return batch[0]

    
    @staticmethod
    def _worker_loop(worker_id, dataset, index_queue, data_queue, collate_fn, done_event, 
                     persistent_workers=False):
        torch.set_num_threads(1)
        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                index = index_queue.get(timeout=3.0)
            except queue.Empty:
                if done_event.is_set():
                    break
                continue
            if done_event.is_set():
                break
            batch = collate_fn([dataset[i] for i in index])
            data_queue.put((batch, worker_id, len(index)))
        # data_queue.join()


    def __next__(self):
        return self._try_get_data()
    
    def __del__(self):
        self.shutdown()
        


class DataLoader:
    def __init__(self, dataset, batch_size=1, num_workers=0, samper=None, batch_sampler=None, drop_last=False, 
                 prefetch_factor=2, collate_fn=default_collate, preload=False, preload_device=None, preproc_fun=None, 
                 persistent_workers=False, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.preload = preload
        if self.preload:
            preproc_fun = pin_memory
            if preload_device is None:
                preload_device = torch.device('cuda:0')
            if isinstance(preload_device, str):
                preload_device = torch.device(preload_device)
        self.preload_device = preload_device
        self.preproc_fun = preproc_fun
        self.persistent_workers = persistent_workers
        if batch_sampler is not None:
            assert samper is None and batch_size == 1, "batch_sampler and sampler cannot be set at the same time"
        if samper is None:
            samper = SequentialSampler(dataset)
        if batch_sampler is None:
            batch_sampler = BatchSampler(samper, batch_size, drop_last=drop_last)
        self.batch_sampler = batch_sampler
        self._iterator = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __iter__(self):
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        if self.persistent_workers and self._iterator is not None:
            self._iterator.reset(self)
        elif self._iterator is not None:
            self._iterator.shutdown()
            self._iterator = None
        if self._iterator is None:
            self._iterator = _MultiProcessingDataLoaderIter(self)
        return self._iterator

    def __len__(self):
        if self.batch_sampler.drop_last:
            return len(self.dataset) // self.batch_sampler.batch_size
        return math.ceil(len(self.dataset) / self.batch_sampler.batch_size)


def worker_fn(dataset, index_queue, output_queue):
    while True:
        # Worker function, simply reads indices from index_queue, and adds the
        # dataset element to the output_queue
        try:
            index = index_queue.get(timeout=1)
        except queue.Empty:
            continue
        if index is None:
            break
        output_queue.put((index, dataset[index]))


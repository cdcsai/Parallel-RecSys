from multiprocessing import Array, Process, Lock
from recsys.tools import flatten

import ctypes
import numpy as np
from itertools import groupby


def share_array(base_array, lock=False) :
    base_shape = base_array.shape
    shared_array_base = Array(ctypes.c_double, base_array.flatten(), lock=lock)
    if lock :
        shared_array = np.frombuffer(shared_array_base.get_obj())
    else:
        shared_array = np.ctypeslib.as_array(shared_array_base)
    return shared_array.reshape(base_shape[0], base_shape[1])


def initialize_uv(u_rows, u_cols, v_cols, with_lock=False) :
    U = share_array(base_array=np.random.rand(u_rows, u_cols) * .1, lock=with_lock)
    V = share_array(base_array=np.random.rand(u_cols, v_cols) * .1, lock=with_lock)

    U[0, :] = [1 for _ in range(u_cols)]
    V[:, 0] = [1 for _ in range(u_cols)]

    return U, V


def parallel(fun, args_list, n_processes, with_lock=False):
    if with_lock :
        lock = Lock()
        jobs_args_list = [(lock, [fun, args_list[s]]) for s in range(n_processes)]
        processes = [Process(target=job, args=jobs_args_list[s]) for s in range(n_processes)]
    else:
        processes = [Process(target=fun, args=args_list[s]) for s in range(n_processes)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()


def job(lock, args):
    lock.acquire()
    fun = args[0]
    fun_args = args[1]
    try:
        fun(fun_args[0], fun_args[1])
    finally:
        lock.release()


def count_conflicts(full_idxs):
    flat = flatten(full_idxs)
    flat.sort()
    frequencies = [len(list(group)) for key, group in groupby(flat)]
    return sum(frequencies) - len(frequencies)

from multiprocessing import Array, Process
import ctypes
import numpy as np


def share_array(base_array) :
    base_shape = base_array.shape
    shared_array_base = Array(ctypes.c_double, base_array.flatten(), lock=False)
    shared_array = np.ctypeslib.as_array(shared_array_base)
    return shared_array.reshape(base_shape[0], base_shape[1])


def initialize_uv(u_rows, u_cols, v_cols) :
    U = share_array(base_array=np.random.rand(u_rows, u_cols) * .1)
    V = share_array(base_array=np.random.rand(u_cols, v_cols) * .1)

    U[0, :] = [1 for _ in range(u_cols)]
    V[:, 0] = [1 for _ in range(u_cols)]

    return U, V


def parallel(fun, args_list, n_processes):
    processes = [Process(target=fun, args=args_list[s]) for s in range(n_processes)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
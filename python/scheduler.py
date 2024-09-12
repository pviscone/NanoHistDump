import multiprocessing
import os

import tqdm


def file_splitting(func, args, files, ncpu=os.cpu_count()):
    pool = multiprocessing.Pool(processes=ncpu)
    res = []
    pbar = tqdm.tqdm(total=len(files))
    def update(*a):
        pbar.update()
    for file_idx, file_path in enumerate(files):
        arguments=[]
        for arg in args:
            if arg=="file_path":
                arguments.append(file_path)
            elif arg=="file_idx":
                arguments.append(file_idx)
            else:
                arguments.append(arg)
        res.append(
            pool.apply_async(
                func,
                args=arguments,
                callback=update
            )
        )
    pool.close()
    pool.join()
    return [r.get() for r in res]


# %%

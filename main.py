from itertools import product
from pathlib import Path
from multiprocessing import Process, Pool

from decode import run

PARALLEL = False

def main():
    n_processes = 15

    n_components = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    exps = ['grasp', 'imagine']
    bands = [['beta'], ['high_gamma'], ['beta', 'high_gamma']]
    ppts = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
    
    jobs = product(ppts, exps, bands)

    if PARALLEL:
        
        pool = Pool(processes=n_processes)
        for job in jobs:
            pool.apply_async(run, args=((job)))
        pool.close()
        pool.join()

    else:
        for job in jobs:
            run(*job)

if __name__=='__main__':
    main()
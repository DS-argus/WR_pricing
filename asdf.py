from multiprocessing import Pool
import time

# 반복문을 실행할 함수
def func(i):
    print(i)

if __name__=='__main__':
    st = time.time()
    pool = Pool(processes=5)
    pool.map(func, range(0, 100000))
    print(time.time()-st)
 
import threading
import time
import queue
import numpy as np
 
# 下面来通过多线程来处理Queue里面的任务：
def work(q):
    while True:
        if q.empty():
            return
        else:
            t = q.get()

            # size_a = size_b = 2
            # a = np.random.rand(size_a,size_a)
            # b = np.random.rand(size_b,size_b)
            # x = a @ b

            mat = np.load('/tmp/' + str(t)  + '.npy')

            # print("当前线程sleep {} 秒".format(t))
            # time.sleep(t)
 
 
def main():
    size_a = 4000
    len_q = 100

    q = queue.Queue()
    
    # np_mat = np.array(np.random.rand(size_a,size_a))
    # for i in range(len_q):
    #     np.save('/tmp/' + str(i), np_mat)

    for i in range(len_q):
        q.put(i)  # 往队列里生成消息

    # single thread
    start = time.time()
    work(q)
    print('time elapsed (single thread)：', time.time() - start) 


    # print(q.qsize())

    for i in range(len_q):
        q.put(i)  # 往队列里生成消息
    # multi-threading


    thread_num = 3
    threads = []
    for i in range(thread_num):
        t = threading.Thread(target=work, args=(q,))
        # args需要输出的是一个tuple，如果只有一个参数，后面加，表示元组，否则会报错
        threads.append(t)

    start = time.time()

    for i in range(thread_num):
        threads[i].start()
    for i in range(thread_num):
        threads[i].join()

    print('time elapsed (multi-threading)：', time.time() - start) 
 
if __name__ == "__main__":
    main()

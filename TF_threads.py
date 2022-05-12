# %%

import tensorflow as tf 
import numpy as np
import sys  

# %%
  
#创建的图:一个先入先出队列,以及初始化,出队,+1,入队操作  
q = tf.FIFOQueue(3, "float")  
init = q.enqueue_many(([0.1, 0.2, 0.3],))  
x = q.dequeue()  
y = x + 1  
q_inc = q.enqueue([y])  
  
#开启一个session,session是会话,会话的潜在含义是状态保持,各种tensor的状态保持  
with tf.Session() as sess:  
        sess.run(init)  
  
        for i in range(2):  
                # sess.run(y)  
                print('len_q: ', sess.run(q.size())) 
                sess.run(q_inc)  
  
        quelen =  sess.run(q.size()) 

        for i in range(quelen):  
                print (sess.run(q.dequeue()))
                print('len_q: ', sess.run(q.size())) 


# %%

# 先申明队列
queue = tf.FIFOQueue(2, "float")
# 定义队列的入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])

# 表示创建了5个线程，每个线程中运行的是enqueue_op操作
qr = tf.train.QueueRunner(queue, [enqueue_op] * 2)

# 加入到TensorFlow的计算图中  使用TensorFlow的默认计算图
tf.train.add_queue_runner(qr)

# 定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    # 使用tf.train.Coordinator来协同启动线程
    coord = tf.train.Coordinator()
    # 使用tf.trainQueueRunner 时，需要明确调用tf.train.start_queue_runners来启动所有线程。否则会因为没有线程运行入队操作。
    # 当调用出队操作时，程序会一直等待入队操作被运行。（这里是理解的重点）
    # tf.train.start_queue_runners 函数会默认启动 tf.GraphKeys.QUEUE_RUNNERS集合总所有的QueueRunner。
    # 该函数只支持启动指定集合中的QueueRunner，所以在使用tf.train.add_queue_runner()和tf.train.start_queue_runners
    # 时会指定同一个集合。
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    # 获取队列中的取值
    for _ in range(10):
        print(sess.run(out_tensor)[0])

    # 使用tf.train.Coordinator来停止所有线程
    coord.request_stop()
    coord.join()


# %%

q = tf.FIFOQueue(10, "float")  
counter = tf.Variable(0.0)  #计数器
# 给计数器加一
increment_op = tf.assign_add(counter, 1.0)
# 将计数器加入队列
enqueue_op = q.enqueue(counter)
 
# 创建QueueRunner，用多个线程向队列添加数据
# 这里实际创建了4个线程，两个增加计数，两个执行入队
qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op])
 
# 主线程
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 启动入队线程
qr.create_threads(sess, start=True)
for i in range(20):
    print(q)
    print(sess.run(q.dequeue()))
    print(counter.eval())

# %%
import tensorflow as tf

# 1000个4维输入向量，每个数取值为1-10之间的随机数
data = 10 * np.random.randn(1000, 4) + 1
# 1000个随机的目标值，值为0或1
target = np.random.randint(0, 2, size=1000)

# 创建Queue，队列中每一项包含一个输入数据和相应的目标值
queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.int32], shapes=[[4], []])

# 批量入列数据（这是一个Operation）
enqueue_op = queue.enqueue_many([data, target])
# 出列数据（这是一个Tensor定义）
data_sample, label_sample = queue.dequeue()

# 创建包含4个线程的QueueRunner
qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)

with tf.Session() as sess:
    # 创建Coordinator
    coord = tf.train.Coordinator()
    # 启动QueueRunner管理的线程
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    # 主线程，消费100个数据
    for step in range(200):
        if coord.should_stop():
            break
        print(sess.run([data_sample, label_sample]))
        # data_batch, label_batch = sess.run([data_sample, label_sample])
    # 主线程计算完成，停止所有采集数据的进程
    coord.request_stop()
    coord.join(enqueue_threads)

# %%
'''FIFO队列操作'''

# 创建队列
# 队列有两个int32的元素

q = tf.FIFOQueue(10,'int32')

# 初始化队列
init= q.enqueue_many(([0,10],))

# 出队
x = q.dequeue()
y = x + 1

# 入队
q_inc = q.enqueue([y])


with tf.Session() as sess:

    init.run()
    print(sess.run(q.size()))

    init.run()
    print(sess.run(q.size()))

    v, _ = sess.run([x, q_inc])

    v = sess.run([x])
    print(v, sess.run(q.size()))

    v = sess.run([x])
    print(v, sess.run(q.size()))

    v = sess.run([x])
    print(v, sess.run(q.size()))

    v = sess.run([x])
    print(v, sess.run(q.size()))

    # Now our queue is empty, if we call it again, our program will hang right here
    # waiting for the queue to be filled by at least one more datum

    sess.run(q.close(cancel_pending_enqueues=True))

    print(sess.run(q.is_closed()))
    
# %%
import tensorflow as tf

# This time, let's start with 6 samples of 1 data point
x_input_data = tf.random_normal([1], mean=0, stddev=4)

# Note that the FIFO queue has still a capacity of 3
q = tf.FIFOQueue(capacity=1, dtypes=tf.float32)

enqueue_op = q.enqueue_many(x_input_data)

# To leverage multi-threading we create a "QueueRunner"
# that will handle the "enqueue_op" outside of the main thread
# We don't need much parallelism here, so we will use only 1 thread
numberOfThreads = 1 
qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)
# Don't forget to add your "QueueRunner" to the QUEUE_RUNNERS collection
tf.train.add_queue_runner(qr) 

input = q.dequeue() 
# input = tf.Print(input, data=[q.size(), input], message="Nb elements left, input:")

# fake graph: START
y = input + 1
# fake graph: END 

# We start the session as usual ...
with tf.Session() as sess:
    # But now we build our coordinator to coordinate our child threads with
    # the main thread
    coord = tf.train.Coordinator()
    # Beware, if you don't start all your queues before runnig anything
    # The main threads will wait for them to start and you will hang again
    # This helper start all queues in tf.GraphKeys.QUEUE_RUNNERS
    threads = tf.train.start_queue_runners(coord=coord)

    # The QueueRunner will automatically call the enqueue operation
    # asynchronously in its own thread ensuring that the queue is always full
    # No more hanging for the main process, no more waiting for the GPU
    print(sess.run([y, q.size()]))
    print(sess.run([y, q.size()]))
    print(sess.run([y, q.size()]))
    print(sess.run([y, q.size()]))
    print(sess.run([y, q.size()]))


    # We request our child threads to stop ...
    coord.request_stop()
    # ... and we wait for them to do so before releasing the main thread
    coord.join(threads)
# %%

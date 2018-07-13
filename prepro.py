
import os
import time
from scapy.all import *
#from pcapng.scanner import FileScanner
import numpy as np
import os
import multiprocessing as mp
import pickle as pk
from utils import *



with open('objs/fileName2Application.pickle', 'rb') as f:
    dict_name2label = pk.load(f)


with open('objs/fileName2Characterization.pickle', 'rb') as f:
    dict_name2class = pk.load(f)


def pkts2X(pkts):
    X = []
    #lens = []
    for p in pkts:
        #===================================
        # step 1 : remove Ether Header
        #===================================
        r = raw(p)[14:]
        r = np.frombuffer(r, dtype = np.uint8)
        #p.show()
        #===================================
        # step 2 : pad 0 to UDP Header
        # it seems that we need to do nothing this step
        # I found some length of raw data is larger than 1500
        # remove them.
        #===================================
        if (TCP in p or UDP in p):
            """
            if UDP in p:
                # todo : padding 0 to 
                print ('UDP', r[:20])
                print(p[IP].src, p[IP].dst)
            else :
                print('TCP', r[:20])
                print(p[IP].src, p[IP].dst)
            """
            if (len(r) > 1500):
                pass
            else:
                X.append(r)
                #lens.append(len(r))
        else:
            pass
    return X#, lens



def get_data_by_file(filename):
    pkts = rdpcap(filename)
    X = pkts2X(pkts)
    # save X to npy and delete the original pcap (it's too large).
    return X

def task(filename):
    global dict_name2label
    global counter
    head, tail = os.path.split(filename)
    cond1 = os.path.isfile(os.path.join('data', tail+'.pickle'))
    cond2 = os.path.isfile(os.path.join('data', tail+'_class.pickle'))
    if (cond1 and cond2):
        with lock:
            counter.value += 1        
        print('[{}] {}'.format(counter, filename))
        return '#ALREADY#'
    X = get_data_by_file(filename)
    if (not cond1):
        y = [dict_name2label[tail]] * len(X)
        with open(os.path.join('data', tail+'.pickle'), 'wb') as f:
            pk.dump((X, y), f)
    if (not cond2):
        y2 = [dict_name2class[tail]] * len(X)
        with open(os.path.join('data', tail+'_class.pickle'), 'wb') as f:
            pk.dump(y2, f)
    with lock:
        counter.value += 1
    print('[{}] {}'.format(counter, filename))
    return 'Done'


#=========================================
# mp init
#=========================================
lock = mp.Lock()
counter = mp.Value('i', 0)
cpus = mp.cpu_count()//2
pool = mp.Pool(processes=cpus)



todo_list = gen_todo_list('../pcaps')

#todo_list = todo_list[:3]

total_number = len(todo_list)

done_list = []
    
res = pool.map(task, todo_list)

print(len(res))



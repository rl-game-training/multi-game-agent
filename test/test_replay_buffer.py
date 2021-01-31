import sys
sys.path.insert(0, "../src")
from train_assault import Buffer
from copy import deepcopy

def print_buf(buf):

    print(buf.storage)

if __name__ == '__main__':

    buf = Buffer(4)

    for i in range(10):


        if(len(buf.storage) < buf.capacity):
            buf.insert(i)
            continue

        next_el_buf = deepcopy(buf)
        next_el_buf.insert(50+i)
        buf.insert(i)

        print("old buf:")
        print_buf(buf)
        print("new buf:")
        print_buf(next_el_buf)


        

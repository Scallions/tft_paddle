import os
from datetime import datetime



def get_timestamp(HMS=False):
    if HMS:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        return datetime.now().strftime("%Y-%m-%d")

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def write_log(msg,path,name):
    time = get_timestamp(HMS=True)
    YMD = get_timestamp(HMS=False)
    MSG = time + msg
    create_dir(path)
    save_path = os.path.join(path,name+'_'+YMD+'.log')
    with open(save_path,'a') as f:
        f.write(MSG+'\n')

if __name__ == '__main__':
    a = get_timestamp(HMS=True)
    print(a)
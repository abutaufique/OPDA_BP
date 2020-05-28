import numpy as np
import pdb
def loadtxt(filename):
    return np.loadtxt(filename, dtype=str)
def seperator(da, wso):
    return ~np.isin(wso, da)
def Dump_text(files, filename):
    with open(filename,'w') as fw:
        for item in files:
            fw.write(item[0] + ' ' + str(item[1]) + ' ' + str(item[2]) + '\n')

if __name__ == '__main__':
    da = loadtxt('opda_failure.txt')
    os = loadtxt('opda_failure_os.txt')
    da_success = seperator(da[:,0], os[:,0])
    da_success = os[da_success]
    da_success = [[item[0], int(item[1]), int(item[2])] for item in da_success]
    Dump_text(da_success, 'da_success.txt')

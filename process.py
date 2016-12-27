import sys
import pickle
import numpy as np

def Count(data):
    d = 0.0
    for i in range(0, len(data)):
        if(data[i] == True):
            d = d + 1.0
    return float(d/len(data))

def toList(data):
    (A, B, C, D, E) = np.array(data).shape
    weight = []
    for b in range(0, B):
        for c in range(0, C):
            for d in range(0, D):
                for e in range(0, E):
                    weight.append(data[0][b][c][d][e])
    print(max(weight))
    print(min(weight))
    return weight

def Small2Zero(list_data, n):
    dmax = max(list_data)
    dmin = min(list_data)
    stride = (dmax - dmin)/n
    for i in range(0, len(list_data)):
        tmp = list_data[i];
        if ((tmp>(dmin+n/2*stride)) and (tmp<(dmin+(n/2+1)*stride))):
            list_data[i] = 0.0;
    return list_data

def Reshape(data, list_data):
    (A, B, C, D, E) = np.array(data).shape
    i = 0
    for b in range(0, B):
        for c in range(0, C):
            for d in range(0, D):
                for e in range(0, E):
                    data[0][b][c][d][e] = list_data[i]
                    i = i + 1 
    return data

# element wise substraction ( data2 - data1 )
def Diff(data1, data2):
    (A, B, C, D, E) = np.array(data1).shape
    data = []
    data.append([])
    i = 0
    for b in range(0, B):
        data[0].append([])
        for c in range(0, C):
            data[0][b].append([])
            for d in range(0, D):
                data[0][b][c].append([])
                for e in range(0, E):
                    data[0][b][c][d].append(data2[0][b][c][d][e] - data1[0][b][c][d][e])
    return data

# element wise absolute( data1 )
def DataAbs(data1):
    (A, B, C, D, E) = np.array(data1).shape
    data = []
    data.append([])
    for b in range(0, B):
        data[0].append([])
        for c in range(0, C):
            data[0][b].append([])
            for d in range(0, D):
                data[0][b][c].append([])
                for e in range(0, E):
                    data[0][b][c][d].append(abs(data1[0][b][c][d][e]))
    return data

# element wise max(data1, data2)
def DataMax(data1, data2):
    (A, B, C, D, E) = np.array(data1).shape
    data = []
    data.append([])
    for b in range(0, B):
        data[0].append([])
        for c in range(0, C):
            data[0][b].append([])
            for d in range(0, D):
                data[0][b][c].append([])
                for e in range(0, E):
                    #data[0][b][c][d][e] = max(data1[0][b][c][d][e], data2[0][b][c][d][e])
                    data[0][b][c][d].append(max(data1[0][b][c][d][e], data2[0][b][c][d][e]))
    return data

# element wise min(data1, data2)
def DataMin(data1, data2):
    (A, B, C, D, E) = np.array(data1).shape
    data = []
    data.append([])
    for b in range(0, B):
        data[0].append([])
        for c in range(0, C):
            data[0][b].append([])
            for d in range(0, D):
                data[0][b][c].append([])
                for e in range(0, E):
                    #data[0][b][c][d][e] = min(data1[0][b][c][d][e], data2[0][b][c][d][e])
                    data[0][b][c][d].append(min(data1[0][b][c][d][e], data2[0][b][c][d][e]))
    return data

# report coordinates of non-zero/zero elements 
# if flag==0 then find coordinates (x, y) of zero values
# else find coordinates (x, y) of non-zero values
def Cordina(flag, data):
    (A, B, C, D, E) = np.array(data).shape
    results = []
    weights = []
    weights.append([])
    for b in range(0, B):
        weights[0].append([])
        for c in range(0, C):
            weights[0][b].append([])
            for d in range(0, D):
                weights[0][b][c].append([])
                for e in range(0, E):
                    if(flag == 'zero'):
                        if(data[0][b][c][d][e] == 0):
                            results.append([b, c, d, e])
                            weights[0][b][c][d].append(0)
                        else:
                            weights[0][b][c][d].append(data[0][b][c][d][e])
                    else:
                        if(data[0][b][c][d][e] != 0):
                            results.append([b, c, d, e])
                            weights[0][b][c][d].append(data[0][b][c][d][e])
                        else:
                            weights[0][b][c][d].append(0)
    return(results, weights)

#++++++++++ EXP 1 ++++++++++++
#1. find and wipe out small values
def exp_1(data, test):
    #---- for test and verify only ----
    if(test):
        tmp = []
        for i in range(0, 5):
            for j in range(0, 5):
                tmp.append(data[0][i][j][0][0])

    #+++ core function +++
    result = Reshape(data, Small2Zero(toList(data), 5))

    #---- for test and verify only ----
    if(test):
        for i in range(0, 5):
            for j in range(0, 5):
                print( str(tmp[i*5+j]) + "\t" + str(result[0][i][j][0][0]) )
    return result

#+++++++++++ EXP 2 ++++++++++++
# 1. find differences between data1 and data2 (data_2 - data_1)
# 2. filter out small values, factor = 5 (histogram of 5, filter out block #3)
def exp_2(data1, data2):
    data = Diff(data1, data2)
    return exp_1(data, 0)

#+++++++++++ EXP 2: common max ++++++++++++
def exp_2_commMax(data1, data2, data3, data4):
    mdata0 = exp_2(data1, data2)
    mdata1 = exp_2(data2, data3)
    mdata2 = exp_2(data3, data4)
    result = DataMax(mdata2, DataMax(mdata0, mdata1))
    cord, weights  = Cordina('non_zero', result)
    return cord, weights

#+++++++++++ EXP 2: common min ++++++++++++
def exp_2_commMin(data1, data2, data3, data4):
    mdata0 = exp_2(data1, data2)
    mdata1 = exp_2(data2, data3)
    mdata2 = exp_2(data3, data4)
    result = DataMin(mdata2, DataMin(mdata0, mdata1))
    cord, weights  = Cordina('zero', result)
    return cord, weights

##----- Main -------
def main(argv):

    arlen = len(argv)
    if(arlen == 2):
        pfile = open(argv[1], 'rb') #load file
        data = pickle.load(pfile)
        pfile.close()
        #print(np.array(data).shape)
        #print(len(Small2Zero(toList(data), 5)))
        result = exp_1(data, 0)
        wf = open('exp1.pkl', 'wb')
        pickle.dump(result, wf)
        wf.close()

    if(arlen == 3):
        print("This is experiment 2") #load file
        p1 = open(argv[1], 'rb')
        p2 = open(argv[2], 'rb')
        data1 = pickle.load(p1)
        data2 = pickle.load(p2)
        p1.close()
        p2.close()

        result = exp_2(data1, data2)
        wf = open('exp2.pkl', 'wb')
        pickle.dump(result, wf)
        wf.close()

    if(arlen == 6):
        print("This is experiment 2: Min or Max Corrdinates")
        p1 = open(argv[1], 'rb')
        p2 = open(argv[2], 'rb')
        p3 = open(argv[3], 'rb')
        p4 = open(argv[4], 'rb')
        data1 = pickle.load(p1)
        data2 = pickle.load(p2)
        data3 = pickle.load(p3)
        data4 = pickle.load(p4)
        p1.close()
        p2.close()
        p3.close()
        p4.close()
        print(argv[5])

        if(argv[5] == 'max'):
            result = exp_2_commMax(data1, data2, data3, data4)
            wf = open('exp2_max.pkl', 'wb')
            pickle.dump(result, wf)
            wf.close()
        if(argv[5] == 'min'):
            result = exp_2_commMin(data1, data2, data3, data4)
            wf = open('exp2_min.pkl', 'wb')
            pickle.dump(result, wf)
            wf.close()


if __name__ == '__main__':
    main(sys.argv)

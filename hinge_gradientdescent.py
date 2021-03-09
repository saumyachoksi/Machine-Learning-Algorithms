import sys
import math
import random

#datafile = sys.argv[1]
datafile = "ionosphere/ionosphere.data"
#datafile = "data.txt"
f = open(datafile)
data = []
i=0
l = f.readline()

#** Read Data **
while(l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    l2.append(1)
    data.append(l2)
    l = f.readline()

rows = len(data)
cols = len(data[0])
f.close()
#print(rows,cols)
#print (data)

#** Read labels **
#labelfile = sys.argv[2]
labelfile = "ionosphere/ionosphere.trainlabels.0"
#labelfile = "labels.txt"
f = open(labelfile)
trainlabels = {}
n = []
n.append(0)
n.append(0)
l = f.readline()

while(l != ''):
    a = l.split()
    if(int(a[0]) == 0):
        trainlabels[int(a[1])] = -1
    else:
        trainlabels[int(a[1])] = int(a[0])
    l = f.readline()
    n[int(a[0])] += 1

#** Initialize w and dellf**
w = []
dellf = []
for j in range(0,cols,1):
#    w.append((0.002*random.uniform(-0.01,0.01))-0.01)
    w.append(random.uniform(-0.01, 0.01))
    dellf.append(0)
print ("Intial W::",w)

def dot(x, y):
    return sum(x_i*y_i for x_i, y_i in zip(x, y))

## Gradient descent iteration

eta = 0.001
theta = 0.001   # Stopping Condition
#theta = 0.001   # Stopping Condition

preverror = float ('inf')
error = 0

for i in range(0, rows, 1):
    if(trainlabels.get(i)!= None):
        #error += ((trainlabels[i] - dot(w,data[i]))**2)
        error += max(0, 1- (trainlabels[i]*dot(w,data[i])))
#print (error)

while (abs(preverror - error) > theta):
    preverror = error
    
    for j in range(0, cols, 1):
        dellf[j] = 0
    
    for i in range(0, rows, 1):
        if(trainlabels.get(i) != None):
            dp = dot(w,data[i])
            for j in range(0, cols, 1):
                if ((trainlabels[i] * dp)<1):
                    dellf[j] += -(data[i][j]*trainlabels[i])
                else:
                    dellf[j] += 0
            #print ("dellf:",dellf)

    for j in range(0, cols, 1):
        w[j] = w[j] - eta * (dellf[j])

    error = 0
    for i in range(0, rows, 1):
        if(trainlabels.get(i)!= None):
            #error += ((trainlabels[i] - dot(w,data[i]))**2)
            error += max(0, 1- (trainlabels[i]*dot(w,data[i])))
    print ("error",error)
    #print ("diff",preverror - error)

print ("Final W::",w)
normw = 0

for j in range(0, cols-1, 1):
    normw += w[j]**2
normw = math.sqrt(normw)

print ("||w||=", normw)
d_origin = w[len(w)-1]/normw

print ("dist to orgin=", abs(d_origin))

for i in range(0, rows, 1):
    if(trainlabels.get(i)==None):
        dp = dot(w,data[i])
        if(dp>0):
            print ("1", i)
        else:
            print ("0", i)

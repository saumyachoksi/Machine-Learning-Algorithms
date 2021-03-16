import sys
#datafile = sys.argv[1]
datafile = "breast_cancer/breast_cancer.data"
#datafile = "data.txt"
f = open(datafile)
data = []
i=0
l = f.readline()

### Read Data
while(l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    data.append(l2)
    l = f.readline()

rows = len(data)
cols = len(data[0])
f.close()
#print(rows,cols)

### Read labels
#labelfile = sys.argv[2]
labelfile = "breast_cancer/breast_cancer.trainlabels.0"
#labelfile = "labels.txt"
f = open(labelfile)
trainlabels = {}
n = []
n.append(0)
n.append(0)
l = f.readline()
while(l != ''):
    a = l.split()
    trainlabels[int(a[1])] = int(a[0])
    l = f.readline()
    n[int(a[0])] += 1
#print(trainlabels)

### Compute means
m0 = []
for j in range(0, cols, 1):
    m0.append(0)

m1 = []
for j in range(0, cols, 1):
    m1.append(0)

for i in range(0, rows, 1):
    if(trainlabels.get(i) != None and trainlabels[i] ==0):
        for j in range(0, cols, 1):
            m0[j] = m0[j] + data[i][j]
    if(trainlabels.get(i) != None and trainlabels[i] ==1):
        for j in range(0, cols, 1):
            m1[j] = m1[j] + data[i][j]


for j in range(0, cols, 1):
    m0[j] = m0[j]/n[0]
    m1[j] = m1[j]/n[1]

#print("Mean")
#print(m0)
#print(m1)

###Compute Variance
v0 = []
for j in range(0, cols, 1):
    v0.append(0)
v1 = []
for j in range(0, cols, 1):
    v1.append(0)

for i in range(0, rows, 1):
    if(trainlabels.get(i) != None and trainlabels[i] ==0):
        for j in range(0, cols, 1):           
            v0[j] = v0[j] + ((data[i][j] - m0[j])**2)        
    if(trainlabels.get(i) != None and trainlabels[i] ==1):
        for j in range(0, cols, 1):           
            v1[j] = v1[j] + ((data[i][j] - m1[j])**2)        

for j in range(1, cols, 1):
    v0[j] = v0[j]/n[0]
    v1[j] = v1[j]/n[1]

#print("Var")
#print(v0)
#print(v1)


### Classify unlabeled points
for i in range(0, rows, 1):
    if(trainlabels.get(i) == None):        
        for j in range(0, cols, 1):
            d0 = 0
            d1 = 0
            for j in range(0, cols,1):
                d0 = d0 + ((( m0[j] - data[i][j])**2)/v0[j])
                d1 = d1 + (((m1[j] - data[i][j])**2)/v1[j])
        if (d0<d1):
            print("0", i)
        else:
            print("1", i)

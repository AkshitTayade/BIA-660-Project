import numpy as np

clusters = [[] for i in range(3)]
print(clusters)

xdata = np.random.randint(1,50,size=(6,2))
centers = np.random.randint(1,50,size=(3,2))

print("xdata")
print(xdata)
print("centers")
print(centers)

t = np.concatenate(tuple([xdata]*3), axis=1)
# print("cat t over 0 reshaped")
t=t.reshape((18,2))
# print(t.T)
c = np.concatenate(tuple([centers]*6))
# print("cat c over 1")
# print(c.T)
# print("distances")
# print(((t-c)**2).T)
# print("summed")
d = np.sum((t-c)**2,axis=1).T
# print(d)
# print("reshaped")
d = d.reshape((6,3))
# print(d)

print(np.argmin(d,axis=1))
# for i in np.argmin(d,axis=1):
#     print(i,end=' ')
print("\n")
print('find')
for doc,cluster in enumerate(np.argmin(d,axis=1)):
    print(cluster)
    print(doc)
    print(xdata[doc])
    clusters[cluster].append(xdata[doc])
print('find')
clusters = np.array(clusters)
for cluster in clusters:
    print(cluster)
print('.')
print(np.mean(clusters, axis=0))
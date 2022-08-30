from sklearn.neighbors import KNeighborsRegressor

# Have a program in the Linear regression algorithm dir better than this
l1 = []
l2 = []
for i in range(1, 1001):
    a = []
    a.append(i)
    l1.append(a)
for j in range(1, 1001):
    b = []
    b.append(j/2)
    l2.append(b)
knn: KNeighborsRegressor = KNeighborsRegressor(2)
knn.fit(l1, l2)
print(knn.predict([[5.642673]]))

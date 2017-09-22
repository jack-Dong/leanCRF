import numpy as np

a = np.array([[1,2,3,4,5]])

print("a",a,a.shape)

b = a[0,0:]
print("b",b,b.shape)

c = a[0,:]
print("c",c,c.shape)

d = a[:,0]
print("d",d,d.shape)

#NumPy Practice by Renz Carillo with freeCodeCamp
import numpy as np

#LESSON 1.1 - NUMPY BASICS
#initializing an array
a = np.array([1,2,3])
print(a)

#intializing 2d array
b = np.array([[9,4,2,1,2,4],[1,5,3,3,5,7]])
print(b)

#get dimension
print(a.ndim)

#get shape/ no. of rows and columns, you could also know the dimension with this
print(a.shape)

#get datatype of numpy array
print(a.dtype)

#get total n amount of value
print(a.nbytes)

#LESSON 1.2 - CHANGING SPECIFIC ELEMENTS, ROWS, COLUMNS, ETC.
a = np.array([[1,2,3,4,5,6,7],
              [8,9,10,11,12,13,14]]) #creating a 2 rows 7 columns array
print(a)
print(a.shape)

#get a specific element [row,column]
print(a[1,5]) #access 2nd row, 6th column

#slicing
print(a[0, :]) #access 1st row and all column | ':' means all
print(a[:, 3]) #access all of the values in column 4

#selecting specifics using 'startindex:endindex:stepsize'
print(a[0, 1:6:2]) #select 1st row, 2nd column up to 7th column with step size of 2

#changing values
a[1,5] = 1000
print(a)

a[:,2] = 5 #changing all values in the 3rd column
print(a)

#3d example
b = np.array([[[1,2,12],[3,4,11]],[[5,6,10],[7,8,9]]])
print(b)
print(b.shape) #result is 2,2,3 which means 2 matrix, 2 rows, and 3 columns

#get specific element in a 3d
print(b[0,1,1]) #access 1st matrix, 2nd row, 2nd column

#LESSON 2.3 - INTIALIZING DIFFERENT TYPES OF ARRAYS
#all zeros matrix
a = np.zeros((2,3,3)) #np.zeros(matrix,rows,columns)
print(a)

#all ones matrix
b = np.ones((4,2,2), dtype="int32") #you could initialize datatypes too just like a normal numpy array
print(b)

#any other number
c = np.full((2,2), 92) #creating a 2x2 matrix with 69 as values
print(c)

#copying the shape of other arrays
d = np.array([[1,2,3,4,5],[6,7,8,9,10]])
e = np.full_like(a, 100) #its like filling variable d with 100
print(e)

#random decimal number
print(np.random.rand(2,2))

#passing the shape of other arrays then fill it with random numbers
print(np.random.random_sample(d.shape))

#random integer values
print(np.random.randint(7, size=(3,3))) #7 is the maximum number, size is the shape

#creating an identity matrix, this is important for linear algebra
print(np.identity(5)) #5 means 5 by 5

#repeating numpy arrays
arr = np.array([[1,2,3]])
r1 = np.repeat(arr,3, axis=0) #copy the rows if axis=0, copy the columns in axis=1
print(r1)

#challenge, my own answer
num = np.ones((5,5))
num[1,1:-1] = 0
num[3,1:-1] = 0
num[2,1:4] = 0
num[2,2] = 9
print(num)

#be careful when copying arrays, they are like automatic referencing, they became an alias
a = np.array([1,2,3])
b = a
b[0] = 100
print(a) #look, only the b must be changed right?!?!??!
#fix this using copy
b = a.copy()

#LESSON 2.4 - MATHEMATICS
a = np.array([1,2,3])
a += 1 #you could also do minus,subtraction,multiplication, etc.
print(a)

#take the trigonometric functions
print(np.cos(a))

#LESSON 2.4A - linear algebra - numpy
a = np.ones((2,3))
print(a)

b = np.full((3,2),2)
print(b)

#common multiplication like a*b won't work since the size of the matrix are different
print(np.matmul(a,b)) #matrix multiplication that doesn't have the same size

#getting the determinant
c = np.identity(3) #we all know the determinant or area of this is 1
print(np.linalg.det(c))

#there are a couple more for trace,svd,eigenvalues,inverse matrix, etc. READ THE DOCUMENTATION

#LESSON 2.4B - statistics with numpy
stats = np.array([[1,2,3],[4,5,6]])
print(stats)

#get the max value
print("\n")
print(np.max(stats, axis = 1)) #getting the max in rows
print(np.min(stats)) #getting the min

#reorganizing arrays
before = np.array([[1,2,3,4],[5,6,7,8]])
print(before)
after = before.reshape((4,2)) #changing the shape but the n amount of value must still remain the same
print(after)

#vertically stacking vectors
v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])
print(np.vstack((v1,v2))) #adding v2 on top of v1

#horizontal stacking vectors
h1 = np.ones((2,4))
h2 = np.zeros((2,2))
print(np.hstack((h1,h2)))

#LESSON 2.5 - MISCELLANEOUS
#load data from file
print("\n")
filedata = np.genfromtxt("data.txt", delimiter = ",")
filedata = filedata.astype("int32") #convert everything into int from float
print(filedata)

#boolean masking and advance indexing | converting values into true or false and with that you can create filters
filter = ((filedata > 50) & (filedata < 1000)) #these two condition must be met to be true
print(filter)

#advance index
a = np.array([1,2,3,4,5,6,7,8,9])
#print(a[[1,2,8]]) #get the value of index 1,2,and 8






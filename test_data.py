import numpy as np 

# define binary digit arrays
zero = [[1,1,1,1],
	   [1,0,0,1],
	   [1,0,0,1],
	   [1,0,0,1],
	   [1,1,1,1]]
zero = np.asarray(zero)

one = [[0,1,1,0],
	   [0,0,1,0],
	   [0,0,1,0],
	   [0,0,1,0],
	   [0,0,1,0]]
one = np.asarray(one)

two = [[0,1,1,0],
	   [0,0,0,1],
	   [0,0,1,0],
	   [0,1,0,0],
	   [1,1,1,1]]
two = np.asarray(two)

three = [[1,1,1,0],
	    [0,0,0,1],
	    [0,1,1,0],
	    [0,0,0,1],
	    [1,1,1,0]]
three = np.asarray(three)

four = [[1,0,1,0],
	   [1,0,1,0],
	   [1,1,1,1],
	   [0,0,1,0],
	   [0,0,1,0]]
four = np.asarray(four)

five = [[0,1,1,1],
	   [0,1,0,0],
	   [0,1,1,1],
	   [0,0,0,1],
	   [1,1,1,1]]
five = np.asarray(five)

six = [[1,1,1,0],
	   [1,0,0,0],
	   [1,1,1,1],
	   [1,0,0,1],
	   [1,1,1,1]]
six = np.asarray(six)

seven = [[1,1,1,1],
	   [0,0,0,1],
	   [0,0,1,0],
	   [0,0,1,0],
	   [0,0,1,0]]
seven = np.asarray(seven)

eight = [[1,1,1,1],
	   [1,0,0,1],
	   [1,1,1,1],
	   [1,0,0,1],
	   [1,1,1,1]]
eight = np.asarray(eight)

nine = [[1,1,1,1],
	   [1,0,0,1],
	   [1,1,1,1],
	   [0,0,0,1],
	   [0,0,0,1]]
nine = np.asarray(nine)


# generate data and labels
data = np.asarray([zero, one, two, three, four, five, six, seven, eight, nine])
labels = [0,1,2,3,4,5,6,7,8,9]
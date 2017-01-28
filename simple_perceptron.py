# Single Layer Perceptron Learning Rule : OR Gate Example 

# https://blog.dbrgn.ch/2013/3/26/perceptrons-in-python/
from random import choice
from numpy import array, dot, random
from pylab import plot, ylim


# unit step as an activation function by using lambda

unit_step = lambda x: 0 if x < 0 else 1


# training data as :  x1 x2(inputs)  b(bias) O(output)


training_data = [
    (array([0,0,1]), 0),
    (array([0,1,1]), 1),
    (array([1,0,1]), 1),
    (array([1,1,1]), 1),
]

# print training_data

# some random initial weights
w = random.rand(3)

# print w

errors = []
eta = 0.2     #learning constant
n = 100       #maximum iterations

for i in xrange(n):
    x, expected = choice(training_data)
    result = dot(w, x)
    error = expected - unit_step(result)
    errors.append(error)
    w += eta * error * x


# w <- final weights

# Test
for x, _ in training_data:
    # print x
    result = dot(x, w)
    print("{}: {} -> {}".format(x[:2], result, unit_step(result)))

ylim([-1,1])
plot(errors)

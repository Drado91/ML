import numpy as np

STUDENT={'name': 'Dudi,Dvir',
         'ID': ''}

def gradient_check(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """
    fx, grad = f(x) # take x into Quad (f). quad output two variables so the first enters fx and the second enters grad
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index #Enter to ix the current indecies of matrix x.

        ### modify x[ix] with h defined above to compute the numerical gradient.
        ### if you change x, make sure to return it back to its original state for the next iteration.

        ### YOUR CODE HERE:
        xplus = np.array(x)
        xminus = np.array(x)
        xplus[ix] = x[ix] + h
        xminus[ix] = x[ix] - h
        fx_plus = f(xplus)
        fx_minus = f(xminus)
        numeric_gradient = (fx_plus[0] - fx_minus[0]) / (2 * h)
        #raise NotImplementedError
        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numeric_gradient - grad[ix]) / max(1, abs(numeric_gradient), abs(grad[ix]))
        if reldiff > 1e-5:
            print ("Gradient check failed.")
            print ("First gradient error found at index %s" % str(ix))
            print ("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numeric_gradient))
            return
    
        it.iternext() # Step to next index

    print ("Gradient check passed!")

def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print ("Running sanity checks...")
    gradient_check(quad, np.array(123.456))      # scalar test
    gradient_check(quad, np.random.randn(3,))    # 1-D test
    gradient_check(quad, np.random.randn(4,5))   # 2-D test
    print ("")

if __name__ == '__main__':
    # If these fail, your code is definitely wrong.
    sanity_check()

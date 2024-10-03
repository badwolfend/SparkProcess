import numpy as np
import matplotlib.pyplot as plt
import math

def y_2_u(y):
    """
    Convert from y coordinates to u coordinates, where u**2=1-y**2
    :param y:
    :return:
    """
    return np.sqrt(1-y**2)

def doublefact(p):
    """
    Double factorial function for integer p.
    :param p: integer argument
    :return:
    """
    if p == 1 or p == 2:
        return p
    elif p <=0:
        return 1
    else:
        return p*doublefact(p-2)

def gamma(p):
    """
    Gamma integer function or arg p.
    :param p: integer argument
    :return:
    """
    if p <= 1:
        return p
    else:
        return p*gamma(p-1)

def Cnn(x, n):
    """
    The matrix Cnn of Eq. 19 from Li2007.
    :param x:  argument (should be 0.5)
    :param n:  order of inversion approximation
    :return:
    """
    C=np.zeros((n+1, n+1))
    for i in range(0, n+1):
        for j in range(0, n+1):
            C[i, j] = 1./(x+i+j)
    return C

def Bnn(x, n):
    """
    The right (n+1)x(n) part of the matrix inside the determinant matrix of Li2007 Eq. 22.
    :param x: argument (should be 0.5)
    :param n: order of inversion approximation
    :return:
    """
    C=np.zeros((n+1, n))
    for i in range(1, n+2):
        for j in range(0, n):
            C[i-1, j] = 1./(x+i+j)
    return C

def bi(i, y, o, polys, dx):
    """
    bi is eq. 18 from Li2007.  However, I am using pre-computed solution as a function of
    polynomial expansion order "o".  This avoids numerical problems with 1/u when u is 0.
    :param i: The current row (order) of Eq. 22s matrix.  This is the "i" part of bi.
    :param y: The vlue of y we are computing the inversion for.  It is the lower integration limit.
    :param o: The current polynomial order on which the integration is performed on.
    :param polys: The polynomial fitted coefficients for each order "o"
    :return: Integrated value for all polynomial orders (o) at a given inversion order (i)
    """
    u=y_2_u(y)
    if i == 0:
        valInt = 0
        for oi in range(0, o ):
            valInt=valInt+polys[-(oi + 2)]*(u)**(oi)
        ret=valInt
    else:
        valInt = 0
        for oi in range(0, o):
            valInt=valInt+polys[-(oi+2)]*get_int_expansion_mult(y, i, oi+1)
        ret = valInt
    return ret/dx


def bi2(i, ys, Is, dy, integrator='direct'):
    """
    bi2 uses explicit integration to compute eq. 18 integral from Li2007.  So we need to pass the function
    a the interpolated values Is.
    :param i: The current row (order) of Eq. 22s matrix.  This is the "i" part of bi.
    :param ys: The current array of of ys we are computing the inversion for. It excludes y<y[0].
    :param Is: Interpolated values of the function to invert (the intensity of the phase).
    :param dy: spacing of the y values (for integration).
    :param integrator: Specifies which type of integration to use on data in Is.
    :return: Integrated value at a given inversion order (i)
    """
    if i == 0:
        ret=Is[0]
    else:
        f = ys * (ys ** 2 - ys[0] ** 2) ** (i - 1) * Is
        if integrator == 'direct':
            ret = integrate(f, 0, len(ys) - 1, dx=dy)
        elif integrator == 'trap':
            ret = trap2(f, 0, len(ys)-1, dx=dy)
        elif integrator == 'simpsons':
            ret = simpson(f, 0, len(ys)-1, dx=dy)
    return ret


def integrate(f, a, b, dx=1):
    """
    The direct integration of the function f from a to b (indices).
    :param f: function to integrate
    :param a: lower index of function f to begin integration
    :param b: upper index of function f to end integration
    :param dx: the spacing between grid points.
    :return: integral of f from a to b.
    """
    s0 = 0.5*(f[a] + f[b])
    n = b-a
    s = sum(f[1:n])+s0
    # for i in range(1,n,1):
    #     s = s + f[a + i]
    return dx*s

def trap2(f, a, b, dx=1):
    """
    The trapezoidal integration of the function f from a to b (indices).
    :param f: function to integrate
    :param a: lower index of function f to begin integration
    :param b: upper index of function f to end integration
    :param dx: the spacing between grid points.
    :return: integral of f from a to b.
    """
    n = b - a

    if n-1<= 0 :
        s=0
    else:
        n = b-a
        h = (b - a) / (n - 1)
        s = (h / 2) * (f[0] + 2 * sum(f[1:n - 1]) + f[n - 1])

    return dx*s

def simpson(f, a, b, dx):
    """
    The trapezoidal integration of the function f from a to b (indices).
    :param f: function to integrate
    :param a: lower index of function f to begin integration
    :param b: upper index of function f to end integration
    :param dx: the spacing between grid points.
    :return: integral of f from a to b.
    """
    n = int(b) - int(a)
    h=1
    k=0.0
    x=a + h
    for i in range(1,n//2 + 1):
        k += 4*f[x]
        x += 2*h

    x = a + 2*h
    for i in range(1,n//2):
        k += 2*f[x]
        x += 2*h
    return (h/3)*(f[a]+f[b]+k)*dx


def simpsons(f, a, b, dx=1):
    """
    The trapezoidal integration of the function f from a to b (indices).
    :param f: function to integrate
    :param a: lower index of function f to begin integration
    :param b: upper index of function f to end integration
    :param dx: the spacing between grid points.
    :return: integral of f from a to b.
    """
    n = b - a
    h = 1
    if n-1<= 0 :
        s=f[b]
    else:
        # h = (b - a) / (n -1 )
        s = (h / 3) * (f[0] + 2 * sum(f[:n - 2:2])+ 4 * sum(f[1:n - 1:2]) + f[n - 1])

    return dx*s

def get_int_expansion_mult(y, i, o):
    """
    Here I have symbolically integrated the first column of the matrix in Eq. 22 from y to 1.
    Specifically, the integrations in Eq. 18 have been done symbolically and then they are multiplied by the
    (1/u) and u**(-2n).  We can do this because det(a*A)=a^(n)det(A) for A(nXn).  And if only
    multiplying 1 column of A, then det([au, v])=a*det([u,v]).
    :param y: The vlue of y we are computing the inversion for.  It is the lower integration limit.
    :param i: The current row (order) of Eq. 22s matrix.  This is the "i" part of bi.
    :param o: The current polynomial order on which the integration is performed on.
    :return:  The first column of Eq.22 with outside polynomials of "u" brought inside.
    """
    k=i-1
    ret = (1-y**2)**((o-1)/2)*math.gamma(1+k)*math.gamma(1+o/2)/(2*math.gamma(2+k+o/2))
    return ret

def construct_abel_m(y, n, o, polys, dx=1):
    """
    The main method for inverting the Abel transformed signal (I(y)) to its inverse g(r).
    :param y: The value of y we are computing the inversion for.  It is the lower integration limit.
    :param n: The order of the abel inversion approximation.
    :param o: The order or the polynomial approximation of I.
    :param polys: The fitting constants for the polynomial approximation of I.
    :param dx: This is to convert back from unitless to with units.
    :return: The able inverse transform (g) of I at position y.
    """
    Mout = np.zeros((n+1, n+1))
    columnOne=np.zeros((n+1))

    # First construct the first column
    for r in range(n+1):
        columnOne[r] = bi(r, y, o, polys, dx=dx)*(doublefact(2*r-1)/doublefact(2*r-2))
    Mout[:, 0] = columnOne
    (1 - y ** 2) ** ((o - 1) / 2)
    # Now add to other matrix
    B = Bnn(0.5, n)
    Mout[:, 1:] =B
    return Mout

def check_u(u, delt=0.01):
    """
    Ensure no divide by zero by padding with a small number.
    :param u: coordinate to check against zero.
    :param delt: how close to zero to a number can get.
    :return:
    """
    if type(u) == np.float64:
        if u <= delt:
            u = delt
        if u >= 1-delt:
            u = 1-delt
    elif len(u) > 1:
        u[u<=delt]=delt
        u[u>=1-delt]=1-delt

    return u


def construct_abel_m_notpoly(ys, Is, n, dy, delt=0.01, integrator='direct'):
    """
    The main method for inverting the Abel transformed signal (I(y)) to its inverse g(r).  It uses direct explicit
    integration rather than symbolic as construct_abel_m() uses.
    :param ys: The values of y we are computing the inversion for.  This must be a number from (0, 1).
    :param Is:  The value of the phase at the points in ys.
    :param n:  The order of Li solver to use.
    :param dy: The unitless, normalized step-size for integration.
    :param delt: The lower limit for a number (how close to zero it can get when dividing).
    :param integrator: The type of integration method to use (simpson, trap, direct).
    :return: The able inverse transform (g) of I at position y, however you still need to dividie by det(Cnn) and u.
    """

    Mout = np.zeros((n+1, n+1))
    columnOne=np.zeros((n+1))
    u = check_u(y_2_u(ys[0]), delt=delt)

    # First construct the first column
    for r in range(n+1):
        columnOne[r] = u**(-2*r)*bi2(r, ys, Is, dy=dy, integrator=integrator)*(doublefact(2*r-1)/doublefact(2*r-2))
    Mout[:, 0] = columnOne

    # Now add to other matrix
    B = Bnn(0.5, n)
    Mout[:, 1:] =B
    return Mout

def ex1G(r):
    """
    The example 1 given in Li2007 as well as Shimizu1989, Buie1996, and Chan2006.
    :param r: position from axis of symmetry
    :return: The original emission coefficients at position r.
    """
    return 0.5*(1.0+10*r**2-23*r**4+12*r**6)

def ex1I(y):
    """
    The example 1 given in Li2007 as well as Shimizu1989, Buie1996, and Chan2006.
    :param y: position from axis of symmetry as seen by observer.
    :return: The able inverted intensity seen on the observer side of a cylindrically symmetric
            object.
    """
    u = y_2_u(y)
    return (8/105)*u*(19+34*y**2-125*y**4+72*y**6)

def ex2G(r):
    r=check_u(r, delt=0.0001)
    return (1-r**2)**(-3/2)*np.exp(1.1**2*(1-1/(1-r**2)))

def ex2I(y):
    u = y_2_u(y)
    u=check_u(u, delt=0.0001)
    return np.sqrt(np.pi)/1.1/u*np.exp(1.1**2*(1-1/(1-y**2)))

if __name__ == '__main__':

    #############################################################
    ## Setup some initial values for the inversion ##############
    #############################################################

    ## Select which type of integration to use ##
    explicit = True
    integrator = "direct"

    ## Set order of Abel approximation and polynomial expansion of the phase ##
    n=3     # Orderof Abel approximation
    o = 9   # Order of phase polynomial approximation

    ## Initialize the y (direction perpendicular to axis of symmetry of phase) from 0 to 1 (0 is center) ##
    ys = np.linspace(0,1.0, 100) # evenly spaced points
    dy = ys[1]-ys[0]
    us = y_2_u(ys) # convert to u from y, where u^2 = 1-y^2
    ny = len(ys)

    ## Choose a known testing function to invert (there are two to choose from at the moment) ##
    ##      ex1G() and ex2G(), with the corresponding phase functions ex1I(), and ex2I(). Also
    ##      define whether to add noise
    add_noise=False
    gy = ex2G(ys) # True solution of an analytical phase inversion
    I = ex2I(ys) # Phase that is to be inverted
    if add_noise:
        # Set a target noise power
        target_noise_db = -40
        sigma = np.sqrt(10 ** (target_noise_db / 10))
        mu= 0
        I = I+np.abs(np.random.normal(mu, sigma, len(I)))
    gyA = np.zeros_like(gy)

    ###################################################################
    ## Perform the initial polynomial fitting for the input phase, I ##
    ##      here, fitted = p[0] * u**deg + ... + p[deg]             ###
    ###################################################################
    p_fitted = np.polyfit(us, I, deg=o)
    fitted = np.polyval(p_fitted, us)

    ## Check fit ##
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(ys, I, 'k', linewidth=7, label='I')
    ax.plot(ys, fitted, 'xkcd:red orange', linewidth=4, label='Iy-Fit')
    ax.set_xlabel('y')
    ax.set_ylabel('I')
    ax.legend()
    plt.show()

    #######################################
    ## Now invert I for every value of y ##
    #######################################
    Cnn = Cnn(0.5, n)
    detCnn = np.linalg.det(Cnn)

    if explicit:
        for yi in range(ny - 1):
            M = construct_abel_m_notpoly(ys[yi:], I[yi:], n, dy=dy, delt=0.01, integrator=integrator)
            Mdet = np.linalg.det(M)
            ut = check_u(y_2_u(ys[yi]))
            gyA[yi] = np.linalg.det(M) / check_u(y_2_u(ys[yi]))
        gyA = gyA * (1 / (detCnn))

    else:
        for yi in range(ny ):
            M = construct_abel_m(ys[yi], n, o, p_fitted, dx=1)
            Mdet = np.linalg.det(M)

            gyA[yi] = np.linalg.det(M)
        gyA = gyA * (1 / (detCnn))

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(ys, gy, 'k', linewidth=7, label='Gy-theory')
    ax.plot(ys,gyA, 'xkcd:red orange', linewidth=4, label='Gy-Abel')
    ax.set_xlabel('y')
    ax.set_ylabel('G')
    ax.legend()
    plt.show()
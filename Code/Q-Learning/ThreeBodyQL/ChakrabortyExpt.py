import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sps
from scipy.integrate import odeint
from tqdm import tqdm


def TuylsChakraborty(X, t, C, agentParams):

    T = agentParams

    x = X[0:2]
    y = X[2:4]
    z = X[4:]

    xdot = np.zeros(2)
    ydot = np.zeros(2)
    zdot = np.zeros(2)

    xdot[0] = x[0] * ((C @ z)[0] - np.dot(x, (C @ z))) + T * x[0] * (x[1] * np.log(x[1]/x[0]))
    xdot[1] = x[1] * ((C @ z)[1] - np.dot(x, (C @ z))) + T * x[1] * (x[0] * np.log(x[0]/x[1]))

    ydot[0] = y[0] * ((C @ x)[0] - np.dot(y, (C @ x))) + T * y[0] * (y[1] * np.log(y[1]/y[0]))
    ydot[1] = y[1] * ((C @ x)[1] - np.dot(y, (C @ x))) + T * y[1] * (y[0] * np.log(y[0]/y[1]))

    zdot[0] = z[0] * ((C @ y)[0] - np.dot(z, (C @ y))) + T * z[0] * (z[1] * np.log(z[1]/z[0]))
    zdot[1] = z[1] * ((C @ y)[1] - np.dot(z, (C @ y))) + T * z[1] * (z[0] * np.log(z[0]/z[1]))

    return np.hstack((xdot, ydot, zdot))


if __name__ == '__main__':

    stopCond = False

    while not stopCond:

        S = np.round(np.random.uniform(low=-1, high=8), 2)
        T = np.round(np.random.uniform(low= 0, high=9), 2)

        C = np.array([[1, S], [T, 0]])

        if T > 1:
            if S < 0:
                case = 1
            elif S > 0 and S < 1:
                case = 2
            elif S > 1 and T > S:
                case = 3
            elif S > 1 and T < S:
                case = 4
        elif T > 0 and T < 1:
            if S < 0 and T > S:
                case = 5
            elif S > 0 and S < 1 and T > S:
                case = 6
            elif S > 0 and S < 1 and T < S:
                case = 7
            elif S > 1:
                case = 8
        elif T < 0:
            if S < 0 and T > S:
                case = 9
            elif S < 0 and T < S:
                case = 10
            elif S > 0 and S < 1 and T < S:
                case = 11
            elif S > 1:
                case = 12
        else:
            print('S = {0}, T = {1}'.format(S, T))
            raise Exception("Weird choice of S and T")

        stopCond = (case == 4)

    # print('S = {0}, T = {1}'.format(S, T))

    initConds = [np.array([0.41030259, 0.58969741, 0.56361327, 0.43638673, 0.53308779,
       0.46691221]), np.array([8.00418146e-01, 1.99581854e-01, 2.93321817e-04, 9.99706678e-01,
       1.12708184e-01, 8.87291816e-01]), np.array([0.50203898, 0.49796102, 0.88428137, 0.11571863, 0.03295509,
       0.96704491]), np.array([0.44888546, 0.55111454, 0.34212499, 0.65787501, 0.33387894,
       0.66612106]), np.array([0.92317559, 0.07682441, 0.89499898, 0.10500102, 0.04261427,
       0.95738573]), np.array([0.72500583, 0.27499417, 0.04536833, 0.95463167, 0.60778619,
       0.39221381]), np.array([0.80655268, 0.19344732, 0.91123289, 0.08876711, 0.89704441,
       0.10295559]), np.array([0.97353206, 0.02646794, 0.38587598, 0.61412402, 0.1708327 ,
       0.8291673 ]), np.array([0.46419295, 0.53580705, 0.74290291, 0.25709709, 0.7391003 ,
       0.2608997 ]), np.array([0.75603941, 0.24396059, 0.50732423, 0.49267577, 0.33829168,
       0.66170832])]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('P1')
    ax.set_ylabel('P2')
    ax.set_zlabel('P3')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

                           
                               
    for x0 in initConds:

        tau = 0.75
        agentParams = tau

        t = np.linspace(0, int(1e4), int(1e5) + 1)
        sol = odeint(TuylsChakraborty, x0, t, args=(C, agentParams))
        ax.plot(sol[:, 0], sol[:, 2], sol[:, 4])
        ax.scatter(sol[0, 0], sol[0, 2], sol[0, 4], marker='o', color='r')
        ax.scatter(sol[-1, 0], sol[-1, 2], sol[-1, 4], marker='+', color='k')

    plt.title('Chakraborty Case = {0}. S = {1}, T = {2}'.format(case, S, T))  
  
    plt.show()
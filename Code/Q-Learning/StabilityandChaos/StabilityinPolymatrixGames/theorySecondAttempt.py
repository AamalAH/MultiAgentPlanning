import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def getOrderParams(Xs, dz, Dz):
    q = np.sum(((Xs) ** 2) * Dz)
    dx = np.array([(Xs[i + 1] - Xs[i]) for i in range(len(Xs[:-1]))])
    dxdz = dx / dz
    X = (1 / np.sqrt(q)) * np.sum(dxdz * Dz[:-1])

    return q, X


def lineSearch(xs, rho, b, a, step = 1e-5, R=np.linspace(-0.5, 0.5, num=100)):
    allYs = np.zeros(np.shape(R))
    diff = 1e-7

    for i in range(len(R)):
        xd = xs + diff
        xdm = xs - diff

        fs = getF(xs, rho + R[i], b, a)
        fd = getF(xd, rho + R[i], b, a)
        fdm = getF(xdm, rho + R[i], b, a)

        df = (fd - fdm) / (xd - xdm)

        allYs[i] = abs(1 - np.sum((xs - step * fs/df) * Dz))

    return rho + R[np.argmin(allYs)]


def getF(Xs, rho, b, a):
    q, chi = getOrderParams(Xs, dz, Dz)
    return np.array([getFi(i, zs[i], Xs, rho, b, a, q, chi) for i in range(len(Xs))])


def getFi(xi, z, Xs, rho, b, a, q, chi):
    return (b**2)*G * chi * Xs[xi] - a * np.log(Xs[xi]) + b * np.sqrt(q) * z - rho


def findrho(Xs, zs, b, a):
    q, chi = getOrderParams(Xs, dz, Dz)
    allrho = np.zeros(len(Xs))
    for xi in range(len(Xs)):
        allrho[xi] = (b ** 2) * G * chi * Xs[xi] - a * np.log(Xs[xi]) + b * np.sqrt(q) * zs[xi]

    return np.mean(allrho)

if __name__ == "__main__":

    nAct = 10

    G = -1
    a = 0.1
    b = 0.1 / np.sqrt(nAct)

    allA = np.zeros(10)

    for cI, a in tqdm(enumerate(np.linspace(0, 0.05, num=10))):

        Xs = abs(np.random.rand(100))
        zs = np.linspace(-1.5, 1.5, 100)
        Zz = 1 / (np.sqrt(2 * np.pi)) * np.exp((zs ** 2) / 2)
        dz = zs[1] - zs[0]
        Dz = dz / (np.sqrt(2 * np.pi)) * np.exp((zs ** 2) / 2)
        Xs /= np.sum(Dz * Xs)
        rho = findrho(Xs, zs, b, a)
        allFZero = np.zeros(1000)
        variation = 1e-5

        for cIter in range(1000):
            XSampled = np.vstack((Xs + (1e-4 / (cIter + 0.1)) * np.random.multivariate_normal(np.zeros(100), np.eye(100), size=(99)), Xs))
            allFSampled = np.array([getF(X, rho, b, a) for X in XSampled])
            Xs = abs(XSampled[np.argmin(allFSampled[:, 0])])
            Xs /= np.sum(Xs * Dz)
            rho = findrho(Xs, zs, b, a)
            allFZero[cIter] = getF(Xs, rho, b, a)[50]

        _, chi = getOrderParams(Xs, dz, Dz)

        allA[cI] = np.sum(Dz * 1 / (abs((a / Xs) - b**2 * G * chi) ** 2))


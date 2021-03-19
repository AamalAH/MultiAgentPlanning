def FPOrbit(x0, G, nIter):
    A, B = G
    pX, qY = x0[0][:, 0], x0[1][:, 0]

    print(pX.shape)

    dim = np.shape(pX)[0]

    allPX = np.zeros((dim, nIter))
    allQY = np.zeros((dim, nIter))

    erX, erY = pX.T @ A @ qY, pX.T @ B @ qY
    allerX, allerY = np.zeros(nIter), np.zeros(nIter)

    for n in tqdm(range(1, int(nIter) + 1)):
        eX, eY = np.zeros(dim), np.zeros(dim)
        brX, brY = np.argmax(A @ qY), np.argmax(pX.T @ B)
        eX[brX] = 1
        eY[brY] = 1

        pX = (n * pX + eX) / (n + 1)
        qY = (n * qY + eY) / (n + 1)

        allPX[:, n-1] = pX
        allQY[:, n-1] = qY

        erX = (n * erX + (pX.T @ A @ qY)) / (n + 1)
        erY = (n * erY + (pX.T @ B @ qY)) / (n + 1)

        allerX[n-1] = erX
        allerY[n-1] = erY

    return allPX, allQY, allerX, allerY

def plotOnSimplex(gamma, trajX, trajY):
    f, (ax) = plt.subplots(1, 2)

    proj = np.array(
        [[-1 * np.cos(30 / 360 * 2 * np.pi), np.cos(30 / 360 * 2 * np.pi), 0],
         [-1 * np.sin(30 / 360 * 2 * np.pi), -1 * np.sin(30 / 360 * 2 * np.pi), 1]
         ])

    ts = np.linspace(0, 1, 10000)

    e1 = proj @ np.array([ts, 1 - ts, 0 * ts])
    e2 = proj @ np.array([0 * ts, ts, 1 - ts])
    e3 = proj @ np.array([ts, 0 * ts, 1 - ts])

    ax[0].plot(e1[0], e1[1], 'k-', alpha = 0.3)
    ax[0].plot(e2[0], e2[1], 'k-', alpha = 0.3)
    ax[0].plot(e3[0], e3[1], 'k-', alpha = 0.3)

    ax[1].plot(e1[0], e1[1], 'k-', alpha=0.3)
    ax[1].plot(e2[0], e2[1], 'k-', alpha=0.3)
    ax[1].plot(e3[0], e3[1], 'k-', alpha=0.3)

    for i in range(trajX.shape[1]):
        d = proj @ trajX[:, i, :]
        ax[0].plot(d[0], d[1], '--', alpha=0.6)
        ax[0].scatter(d[0, -1], d[1, -1], marker='+')

        d = proj @ trajY[:, i, :]
        ax[1].plot(d[0], d[1], '--', alpha=0.6)
        ax[1].scatter(d[0, -1], d[1, -1], marker='+')

    plt.title(str(np.round(gamma, 3)))

    plt.show()

    return (f, (ax))

def sampleFromSimplex(dim, nSamples=1.5e3):
    nSamples = int(nSamples)
    return np.random.dirichlet(np.ones(dim), size=nSamples)

def getAllPayoffs(dim, erX, erY, allPX, allQY, A, B):
    allNE = findNE(A, B)
    allNEPayoffs = [(allNE[i][0] @ A @ allNE[i][1], allNE[i][0] @ B @ allNE[i][1]) for i in
                range(len(allNE))]

    samples = sampleFromSimplex(dim, nSamples=1e4)
    aBar = np.max(samples @ A @ allQY[:, -1])
    bBar = np.max(allPX[:, -1].T @ B @ samples.T)

    return allNEPayoffs, (erX[-1], erY[-1]), (aBar, bBar)

def plotPayoffs(dim, gamma, erX, erY, A, B, allPX, allQY, nIter):
    f, (ax) = plt.subplots(1, 2)

    allNEPayoffs, (_, _), (aBar, bBar) = getAllPayoffs(dim, erX, erY, allPX, allQY, A, B)

    ax[0].plot(range(nIter), erX, '--')
    ax[1].plot(range(nIter), erY, '--')
    for i in range(len(allNEPayoffs)):
        ax[0].plot(range(nIter), [allNEPayoffs[i][0]]*nIter, 'k-')
        ax[1].plot(range(nIter), [allNEPayoffs[i][1]]*nIter, 'k-')

    plt.title(str(np.round(gamma, 3)))

    plt.show()


def findNE(A, B):
    rps = nash.Game(A, B)
    eqs = rps.support_enumeration()
    return list(eqs)


def writeData(dataNo, gamma, dim, erX, erY, allPX, allQY, A, B, nIter):
    plotPayoffs(dim, gamma, erX, erY, A, B, allPX, allQY, nIter)

    allNEPayoffs, rewards, bars = getAllPayoffs(dim, erX, erY, allPX, allQY, A, B)

    data = {'data': dataNo,
            'gamma': np.round(gamma, 3),
            'allNEPayoffsX': [allNEPayoffs[i][0] for i in range(len(allNEPayoffs))],
            'rewardX': rewards[0],
            'ABar': bars[0],
            'allNEPayoffsY': [allNEPayoffs[i][1] for i in range(len(allNEPayoffs))],
            'rewardY': rewards[1],
            'BBar': bars[1],
            'A': A,
            'B': B
            }
    with open('ExperimentData/Data.txt', 'a') as file:
        file.write(str(data) + '\n')
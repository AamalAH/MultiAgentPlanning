import numpy as np


def DMD_RRR(snapshotsX, snapshotsY, eps):
    Dx = np.diag(np.linalg.norm(snapshotsX, axis=0))

    Xm1 = snapshotsX @ np.linalg.pinv(Dx)
    Ym1 = snapshotsY @ np.linalg.pinv(Dx)

    U, S, V = np.linalg.svd(Xm1)
    allI = S[np.where(S > S[0] * eps)]
    k = np.argmax(allI)

    Uk = U[:, :k]
    Vk = V[:, :k]
    Sigk = np.eye(k) * S[:k]

    Bk = Ym1 @ (Vk @ np.linalg.inv(Sigk))

    Q, R = np.linalg.qr(np.hstack((Uk, Bk)))
    Sk = np.diag(R)[:k] @ R[:k, k+1:2*k]

    Lambdak = np.linalg.eig(Sk)

    for i in range(k):
        [sigmaL, wL] =

    pass

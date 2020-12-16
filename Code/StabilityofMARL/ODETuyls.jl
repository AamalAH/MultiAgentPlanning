using DifferentialEquations
using Plots
using LinearAlgebra
using PyCall

function Tuyls!(du, u, p, t)
    α, τ, A, B, nA = p
    x, y = u[1:nA], u[nA + 1:end]
    
    dx, dy = du[1:nA], du[nA + 1:end]
    
    for i=1:nA
        dx[i] = x[i] * α * τ * ((A * y)[i] - dot(x, A * y)) + x[i] * α * sum([x[j] * log(x[j]/x[i]) for j=1:nA])
        dy[i] = y[i] * α * τ * ((B * x)[i] - dot(y, B * x)) + y[i] * α * sum([y[j] * log(y[j]/y[i]) for j=1:nA])
    end
    
#     du[1] = x1 * α * τ * ((A*[y1;y2])[1] - dot([x1;x2], A * [y1;y2])) + x1 * α * x2 * log(x2/x1)
#     du[2] = x2 * α * τ * ((A*[y1;y2])[2] - dot([x1;x2], A * [y1;y2])) + x2 * α * x1 * log(x1/x2)
#     du[3] = y1 * α * τ * ((B*[x1;x2])[1] - dot([y1;y2], B * [x1;x2])) + y1 * α * y2 * log(y2/y1)
#     du[4] = y2 * α * τ * ((B*[x1;x2])[2] - dot([y1;y2], B * [x1;x2])) + y2 * α * y1 * log(y1/y2)

    du = [dx; dy]
    
end

py"""
def generateGames(gamma, nSim, nAct):
    import numpy as np

    nElements = nAct ** 2  # number of payoff elements in the matrix

    cov = np.eye(2 * nElements)  # <a_ij^2> = <b_ji^2> = 1
    cov[:nElements, nElements:] = np.eye(nElements) * gamma  # <a_ij b_ji> = Gamma
    cov[nElements:, :nElements] = np.eye(nElements) * gamma

    rewardAs, rewardBs = np.eye(nAct), np.eye(nAct)

    for i in range(nSim):
        rewards = np.random.multivariate_normal(np.zeros(2 * nElements), cov=cov)

        rewardAs = np.dstack((rewardAs, rewards[0:nElements].reshape((nAct, nAct))))
        # rewardAs = np.dstack((rewardAs, np.array([[1, 5], [0, 3]])))
        rewardBs = np.dstack((rewardBs, rewards[nElements:].reshape((nAct, nAct)).T))
        # rewardBs = np.dstack((rewardBs, np.array([[1, 0], [5, 3]])))
    return [rewardAs[:, :, 1:], rewardBs[:, :, 1:]]
"""

generateGames(gamma, nSim, nAct) = py"generateGames"(gamma, nSim, nAct)

nA = 2
α, τ = [1e-2, 10]

u0 = []
for p = 1:2
    initCond = rand(nA)
    initCond /= sum(initCond)
    u0 = [u0; initCond]
end

tspan = (0.0, 100.0)

A, B = generateGames(-1, 1, nA)

p = (α, τ, A[:, :, 1], transpose(B[:, :, 1]), nA)

prob = ODEProblem(Tuyls!,u0,tspan, p)

sol = solve(prob)
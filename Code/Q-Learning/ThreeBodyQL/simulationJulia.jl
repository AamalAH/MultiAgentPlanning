using LinearAlgebra
using Distributions
using ProgressMeter
using PyPlot

alpha = 0.05
tau = 0.1
gamma = 0.1

agentParams = (alpha, tau, gamma)


function generateMatchingPennies()
    """
    Create Matching Pennies Matrix
    """

    A, B = [1 -1; -1 1], [-1 1; 1 -1]

    return A, B
end

function getActionProbs(qValues, agentParams)
    """
    qValues: nPlayer x nActions x nSim
    return: nPlayer x nActions x nSim
    """
    alpha, tau, gamma = agentParams

    return exp.(tau * qValues)./sum(exp.(tau * qValues), dims=2)
end

function chooseActions(actionProbs)
    return [rand(Bernoulli(actionProbs[p, 1])) + 1 for p = 1:3]
end

function initialiseQ()
    return rand(3, 2)
end

function getRewards(G, bChoice)
    A, B = G

    rewards = zeros(3)
    rewards[1] = A[bChoice[1], bChoice[2]] + B[bChoice[1], bChoice[3]]
    rewards[2] = B[bChoice[1], bChoice[2]] + A[bChoice[2], bChoice[3]]
    rewards[3] = A[bChoice[3], bChoice[1]] + B[bChoice[2], bChoice[3]]

    return rewards
end

function qUpdate!(Q, G, agentParams)

    actionProbs = getActionProbs(Q, agentParams)
    bChoice = chooseActions(actionProbs)
    rewards = getRewards(G, bChoice)

    for i = 1:3
        Q[i, bChoice[i]] += alpha * (rewards[i] - Q[i, bChoice[i]] + gamma * findmax(Q[i, :])[1])
    end

end

function simulate(nIter = 5e3)
    nIter = trunc(Int, nIter)

    G = generateMatchingPennies()
    Q = initialiseQ()

    global firstActionTracker = zeros(3, nIter)

    for cIter = 1:trunc(Int, nIter)
        qUpdate!(Q, G, agentParams)
        firstActionTracker[:, cIter] = getActionProbs(Q, agentParams)[:, 1]
    end

    return firstActionTracker
end

firstActionTracker = simulate()

samples = 25
plot3D(firstActionTracker[1, :], firstActionTracker[2, :], firstActionTracker[3, :])

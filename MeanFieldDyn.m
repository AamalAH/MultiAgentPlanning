%%
clear

global deltaQ;
global deltaT;
global tau;
global eta;

deltaQ = 1e-2;
deltaT = 1e-2;
%% Set up game

tau = 10;
eta = 1;

game = zeros(2, 2, 3);
game(:, :, 1) = [3, 0; 0, 1];
game(:, :, 2) = [3, 5; 5, 1];
nAgents = 1e3;
qValues = rand(2, nAgents);

%% Determine p_init

p = @(Q)  nnz(find(((qValues(1, :) > Q(1)) & (qValues(1, :) < (Q(1) + deltaQ)) & (qValues(2, :) > Q(2)) & (qValues(2, :) < (Q(2) + deltaQ)))))/nAgents;

%% Determine v(t)

v = @(Q, X) eta/(exp(tau * (Q(1))) + exp(tau * (Q(2)))) * [exp(tau * (Q(1))); exp(tau * (Q(2)))] .* (game(:, :, 2) * X - Q);

%% Determine p_dot

pdot = @(Q, X) -1 * sum((p(Q + deltaQ) * v(Q + deltaQ, X) - p(Q) * v(Q, X))./deltaQ);

%% Update p(t)

for t = 0:deltaT:10
   x = [0; 0];            
    for q1 = 0:deltaQ:(1 - deltaQ)
        for q2 = 0:deltaQ:(1 - deltaQ)
            x(1) = x(1) + ((exp(tau * (q1 + deltaQ/2))/(exp(tau * (q1 + deltaQ/2)) + exp(tau * (q2 + deltaQ/2)))) * p([q1; q2]));
            x(2) = x(2) + ((exp(tau * (q2 + deltaQ/2))/(exp(tau * (q1 + deltaQ/2)) + exp(tau * (q2 + deltaQ/2)))) * p([q1; q2]));
        end
    end
   
   xCurrent = x
    
   p_old = p;   
   p = @(Q) p_old(Q) + pdot(Q, xCurrent) * deltaT;
   
   v = @(Q, X) eta/(exp(tau * (Q(1))) + exp(tau * (Q(2)))) * [exp(tau * (Q(1))); exp(tau * (Q(2)))] .* (game(:, :, 1) * X - Q);
   pdot = @(Q, X) -1 * sum((p_old(Q + deltaQ) * v(Q + deltaQ, X) - p_old(Q) * v(Q, X))./deltaQ);
end




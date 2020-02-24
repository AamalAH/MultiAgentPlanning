%%
clear

global deltaQ;
global deltaT;
global tau;
global eta;

deltaQ = 1e-2;
iterQ = 0:deltaQ:(1-deltaQ);
deltaT = 1e-2;
%% Set up game

tau = 10;
eta = 1;

game = zeros(2, 2, 3);
game(:, :, 1) = [3, 0; 0, 1];
game(:, :, 2) = [3, 5; 5, 1];
nAgents = 1e3;
qValues = rand(2, nAgents);


%% determine pInit
p = @(Q)  nnz(find(((qValues(1, :) > Q(1)) & (qValues(1, :) < (Q(1) + deltaQ)) & (qValues(2, :) > Q(2)) & (qValues(2, :) < (Q(2) + deltaQ)))))/nAgents;

pMAT = zeros(1/deltaQ, 1/deltaQ);
for q1 = 1:1/deltaQ
    for q2 = 1:1/deltaQ
        pMAT(q1, q2) = p([iterQ(q1); iterQ(q2)]);
    end
end
%%
x = [0; 0];            
for q1 = 1:1/deltaQ
    for q2 = 1:1/deltaQ
        x(1) = x(1) + ((exp(tau * (iterQ(q1) + deltaQ/2))/(exp(tau * (iterQ(q1) + deltaQ/2)) + exp(tau * (iterQ(q2) + deltaQ/2)))) * pMAT(q1, q2));
        x(2) = x(2) + ((exp(tau * (iterQ(q2) + deltaQ/2))/(exp(tau * (iterQ(q1) + deltaQ/2)) + exp(tau * (iterQ(q2) + deltaQ/2)))) * pMAT(q1, q2));
    end
end

xCurrent = x;

%%

v = @(Q, X) eta/(exp(tau * (Q(1))) + exp(tau * (Q(2)))) * [exp(tau * (Q(1))); exp(tau * (Q(2)))] .* (game(:, :, 1) * X - Q);





pdot = @(Q, X) -1 * sum([(p(Q(1) + 1, Q(2))*v([iterQ(Q(1) + 1); iterQ(Q(2))], X) - p(Q(1), Q(2))*v([iterQ(Q(1)); iterQ(Q(2))], X))/deltaQ; (p(Q(1), Q(2) +1)*v([iterQ(Q(1)); iterQ(Q(2) + 1)], X) - p(Q(1), Q(2))*v([iterQ(Q(1)); iterQ(Q(2))], X))/deltaQ]);
p_old = p;

for q1 = 1:1/deltaQ
    for q2 = 1:1/deltaQ
        if q1 == (1/deltaQ -1)
            
        end
        p(q1, q2) = p(q1, q2) + pdot([q1; q2], xCurrent) * deltaT;
    end
end
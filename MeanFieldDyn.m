%%
clear
%% Set up game
tau = 10;
eta = 1;

game = zeros(2, 2, 3);
game(:, :, 1) = [3, 0; 0, 1];
game(:, :, 2) = [3, 5; 5, 1];
nAgents = 1e4;
qValues = rand(2, nAgents);

%% Determine x(t)

x = [0; 0];
for q1 = 0:deltaQ:(1 - deltaQ)
for q2 = 0:deltaQ:(1 - deltaQ)
p = nnz(find(((qValues(1, :) > q1) & (qValues(1, :) < (q1 + deltaQ)) & (qValues(2, :) > q2) & (qValues(2, :) < (q2 + deltaQ)))))/nAgents;
x(1) = x(1) + ((exp(tau * (q1 + deltaQ/2))/(exp(tau * (q1 + deltaQ/2)) + exp(tau * (q2 + deltaQ/2)))) * p);
x(2) = x(2) + ((exp(tau * (q2 + deltaQ/2))/(exp(tau * (q1 + deltaQ/2)) + exp(tau * (q2 + deltaQ/2)))) * p);
end
end

%% Determine v(t)

v= [0;0];

v(1) = eta * (exp(tau * (q1 + deltaQ/2))/(exp(tau * (q1 + deltaQ/2)))) * (
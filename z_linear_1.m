clear

% Load data for non-life sentences only
data = load('data_nonlife_train.txt');
X = data(:,1:end-1);
y = data(:,end);
[m,n] = size(X);

% Normalize features
[X,~,~] = meanNormalize(X);
X = [ones(m,1) X];  % add bias

% Normal equation for non-life sentences
theta_0 = normalEqnLinear(X,y);
J_ls = computeCostLinear(X,y,theta_0); % best possible train error

% Gradient Descent parameters
% alpha=0.4 is max alpha with iters=10000
alpha = 0.45;
iters = 100;
itheta = zeros(n+1,1);
[theta_gd,J_train] = gradientDescentLinear(X,y,itheta,alpha,iters);

% Display and plot results
T = table(iters,alpha,J_train(end),J_ls,'VariableNames',{'iters','alpha','J_train','J_ls'});
disp('Results')
disp(T)

figure
hold on
plot(1:iters,J_train,'b'); plot([1,iters],[J_ls,J_ls],'--k');
hold off

xlabel('Iterations'); ylabel('J_{train}');
legend('J_{train}','J_{norm eqn}')

% Plot a learning curve for m for non-life sentence data

clear

% Load data for non-life sentences only
data_train = load('data_nonlife_train.txt');
data_cv = load('data_nonlife_cv.txt');
m_train = size(data_train,1);
m_cv = size(data_cv,1);

% Gradient Descent parameters
alpha = 0.2;	% alpha=0.4 is upper end; iters=100 is lower end
iter = 8000;
m_sizes = round(linspace(25,m_train,15)');

% Learning curve for data set size
% [J_train,J_cv,J_ls] = learnCurve_m(data_train,data_cv,alpha,iter,m_sizes);
[J_train,J_cv,~] = learnCurve_m(data_train,data_cv,alpha,iter,m_sizes);

% Plot results
figure
hold on
% plot(m_sizes,J_ls,'--k','LineWidth',1)
plot(m_sizes,J_train,'b','LineWidth',1)
plot(m_sizes,J_cv,'r','LineWidth',1)
hold off

xlabel('Training Set Size (m)'); ylabel('Error (J)');
title({'Learning Curve for m';['\rm\alpha = ',num2str(alpha),'   |   iter = ',num2str(iter)]})
% legend('J_{norm eqn}','J_{train}','J_{cross-val}','Location','southeast')
legend('J_{train}','J_{cross-val}','Location','northeast')



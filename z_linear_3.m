% Plot a learning curve for m for non-life sentence data
% Adds mapFeatures_v2 to get more complex model

clear; close all;

% Load data for non-life sentences only
data_train = load('data_nonlife_train.txt');
data_val = load('data_nonlife_cv.txt');
m_train = size(data_train,1);
m_cv = size(data_val,1);

% Gradient Descent parameters
alpha = 0.1;	% alpha=0.4 is upper end; iters=100 is lower end
iter = 500;
n_sizes = 15;
sizes = round(linspace(25,m_train,n_sizes)');
deg = 5;	% degree of polyFeatures

% Learning curve for data set size
[J_train,J_val,~] = learnCurve_m3(data_train,data_val,alpha,iter,sizes,deg);

% Plot results
figure
hold on
% plot(m_sizes,J_ls,'--k','LineWidth',1)
plot(sizes,J_train,'b','LineWidth',1)
plot(sizes,J_val,'r','LineWidth',1)
hold off

xlabel('Training Set Size (m)'); ylabel('Error (J)');
title({'Learning Curve for m';['\rm\alpha = ',num2str(alpha),'   |   iter = ',num2str(iter)]})
% legend('J_{norm eqn}','J_{train}','J_{cross-val}','Location','southeast')
legend('J_{train}','J_{val}','Location','northeast')



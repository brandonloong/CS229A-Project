% Run NN for non-life sentences�

clear; %close all;

% Load data for non-life sentences only
data_train = load('data_nonlife_train.txt');
data_val = load('data_nonlife_cv.txt');
data_test = load('data_nonlife_test.txt');
m_train = size(data_train,1);

% Gradient Descent parameters
alpha = 0.15;	% alpha=0.4 is upper end; iters=100 is lower end
iter = 5000;
n_sizes = 15;
deg = 2;        % degree of polyFeatures
% sizes = round(linspace(25,m_train,n_sizes)');
sizes = round(logspace(log10(50),log10(m_train),n_sizes)');

% Learning curve for data set size
[J_train,J_val,~,theta,y_model] = learnCurve_m5(data_train,data_val,data_test,alpha,iter,sizes,deg);

% Plot Parity: model vs test
figure
plot(y_model,data_test(:,end),'.')
xlabel('Model Y'); ylabel('Test Y'); axis([0 1200 0 1200]);
title('Parity Plot (data vs model)');

% Plot Results: J_train and J_val
figure
hold on
% plot(m_sizes,J_ls,'--k','LineWidth',1)
plot(sizes,J_train,'b','LineWidth',1)
plot(sizes,J_val,'r','LineWidth',1)
hold off
xlabel('Training Set Size (m)'); ylabel('Error (J)');
title({'Learning Curve for m';...
    ['\rm\alpha = ',num2str(alpha),', iter = ',num2str(iter),', degree = ',num2str(deg)]})
% legend('J_{norm eqn}','J_{train}','J_{cross-val}','Location','southeast')
legend('J_{train}','J_{val}','Location','northeast')



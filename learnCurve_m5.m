function [J_train,J_val,J_test,theta,y_model] = learnCurve_m5(data_t,data_val,data_test,alpha,iter,sizes,deg)
% Create learning curves from the data
% Uses: convertCategory, no normalization or polyFeatures, use
% categorical & numerical data separately.

n_num = 1;
n_sizes = length(sizes);
f = waitbar(0,'Running code...');

% Setup validation data (it doesn't change)
X_v = data_val(:,1:end-1);    % import validation data
y_v = data_val(:,end);
X_val = processX(X_v,deg,n_num);    % pre-process X data

% Setup test data (it doesn't change)
X_test0 = data_test(:,1:end-1);    % import validation data
y_test = data_test(:,end);
X_test = processX(X_test0,deg,n_num);    % pre-process X data

[J_train,J_val,J_ls,J_test] = deal(zeros(n_sizes,1));   % initialize J's
for i=1:n_sizes
    % Randomly pick training set for set i between [1,m_train]
% 	[X_t,y_t] = randomData(data_t,sizes(i));    % random rows each subset
    
    % Sequential rows for each subset
    X_t = data_t(1:sizes(i),1:end-1);
    y_t = data_t(1:sizes(i),end);
    
    % Pre-process X data
    X_train = processX(X_t,deg,n_num);
	n_train = size(X_train,2)-1;		% get n training features
    
    % Normal Equation for set i
    theta_0 = normalEqnLinear(X_train,y_t);
    J_ls(i) = computeCostLinear(X_train,y_t,theta_0); % best possible J_train
%     J_ls(i) = 0;
    
    % Train parameters using gradient descent
    itheta = zeros(n_train+1,1);
    [theta,J_hist] = gradientDescentLinear(X_train,y_t,itheta,alpha,iter);
    J_train(i) = J_hist(end);
    
    % Validate model
    J_val(i) = computeCostLinear(X_val,y_v,theta);  % with GD
%     J_val(i) = computeCostLinear(X_val,y_v,theta_0);  % with NE
	
	% Test model
    J_test(i) = computeCostLinear(X_test,y_test,theta);  % with GD
	
    waitbar(i/n_sizes,f);   % update waitbar
end
close(f);   % close waitbar

% Test model
X_test = data_test(:,1:end-1);
% y_test = data_test(1:sizes(i),end);
X_model = processX(X_test,deg,n_num);
y_model = X_model*theta;
    
% Results
T = table(sizes,J_ls,J_train,J_val,J_test);
disp('Results')
disp(T)

end
function [J_train,J_cv,J_ls] = learnCurve_m(data_train,data_cv,alpha,iter,m_sizes)
% Create learning curves from the data

m_train = size(data_train,1);
n_sizes = length(m_sizes);

[J_train,J_cv,J_ls] = deal(zeros(n_sizes,1));   % initialize J's
f = waitbar(0,'Running code...');
for i=1:n_sizes
    % Get proper data set size
    n_train = m_sizes(i);
    
    % Randomly pick training rows for set i between [1,m_train]
    rand_rows = round(rand(n_train,1)*(m_train-1)) + 1; % add 1 for 0 ind    
    t_set = data_train(rand_rows,:);
    X_t = t_set(:,1:end-1);
    y_t = t_set(:,end);
    
    
% 	isnan(X_t);
% 	sum(sum(isnan(X_t)));
% 	[X_t(:,2:end),~,~] = featureNormalize(X_t(:,2:end));	% normalize features
	[X_t,~,~] = featureNormalize(X_t);	% normalize features
	sum(sum(isnan(X_t)));
	X_t = mapFeature_v2(X_t);			% map features (adds bias)
% 	sum(sum(isnan(X_t)));
	n_t = size(X_t,2)-1;
%     X_t = [ones(m_t,1) X_t];			% add bias
    
    % Normal Equation for set i
%     theta_0 = normalEqnLinear(X_t,y_t);
%     J_ls(i) = computeCostLinear(X_t,y_t,theta_0); % best possible J_train
    J_ls(i) = 0;
    
    % Gradient Descent parameters
    itheta = zeros(n_t+1,1);
    [theta,J_hist] = gradientDescentLinear(X_t,y_t,itheta,alpha,iter);
    J_train(i) = J_hist(end);
    
    % Check CV data
    X_cv = data_cv(:,1:end-1);
    y_cv = data_cv(:,end);
    
	X_cv = mapFeature_v2(X_cv);				% map features (adds bias)
    [X_cv(:,2:end),~,~] = featureNormalize(X_cv(:,2:end));	% normalize features
	
% 	[m_cv,~] = size(X_cv);
%     X_cv = [ones(m_cv,1) X_cv];			% add bias
    
    J_cv(i) = computeCostLinear(X_cv,y_cv,theta);
    
    waitbar(i/n_sizes,f);   % update waitbar
end
close(f);   % close waitbar

% Results
T = table(m_sizes,J_ls,J_train,J_cv);
disp('Results')
disp(T)

end
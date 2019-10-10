function [J_train,J_cv,J_ls] = learnCurve_m2(data_train,data_cv,alpha,iter,sizes)
% Create learning curves from the data

n_sizes = length(sizes);
f = waitbar(0,'Running code...');

[J_train,J_cv,J_ls] = deal(zeros(n_sizes,1));   % initialize J's
for i=1:n_sizes
    % Randomly pick training rows for set i between [1,m_train]
	[X_t,y_t] = randomData(data_train,sizes(i));
    
    % Map features before normalizing
	X_tp = mapFeature_v2(X_t);				% map poly features (no bias)
	bad_cols = find(range(X_tp,1)==0);		% find non-variant features
	X_tp = X_tp(:,~bad_cols);				% remove non-variant features
	[X_tpn,~,~] = featureNormalize(X_tp);	% normalize features
	X_tpn = [ones(size(X_tpn,1),1) X_tpn];	% add bias
	n_t = size(X_tpn,2)-1;					% get n training features
    
    % Normal Equation for set i
%     theta_0 = normalEqnLinear(X_t,y_t);
%     J_ls(i) = computeCostLinear(X_t,y_t,theta_0); % best possible J_train
    J_ls(i) = 0;
    
    % Train parameters using gradient descent
    itheta = zeros(n_t+1,1);
    [theta,J_hist] = gradientDescentLinear(X_tpn,y_t,itheta,alpha,iter);
    J_train(i) = J_hist(end);
    
    % Check CV data
    X_cv = data_cv(:,1:end-1);
    y_cv = data_cv(:,end);
    
	% Map features before normalizing
	X_cvp = mapFeature_v2(X_cv);				% map features (adds bias)
	X_cvp = X_cvp(:,~bad_cols);					% remove non-variant features
    [X_cvpn,~,~] = featureNormalize(X_cvp);		% normalize features
    X_cvpn = [ones(size(X_cvpn,1),1) X_cvpn];	% add bias
    
    J_cv(i) = computeCostLinear(X_cvpn,y_cv,theta);
    
    waitbar(i/n_sizes,f);   % update waitbar
end
close(f);   % close waitbar

% Results
T = table(sizes,J_ls,J_train,J_cv);
disp('Results')
disp(T)

end
function X_total = processX(X,deg,n_num)
%processX Prep data for training model
%   Take original feature data,  separate categorical and numerical data.
%   Convert cat. feature vectors into matrices.
%   PolyFeatures on num. feature vectors, then normalize.
%   Combine both of these results into one X_total
n_cat = size(X,2)-n_num;
X_t_catv = X(:,1:n_cat);      % select category data
X_t_num = X(:,n_cat+1:end);   % select number data

% Category data: convert categories into matrices
X_t_catm = convertCategory(X_t_catv);

% Numerical data: map numerical features before normalizing
X_t_map = polyFeatures(X_t_num,deg);	% make poly features (no bias)
[X_t_mapn,~,~] = featureNormalize(X_t_map);	% normalize features

X_total = [X_t_catm X_t_mapn];  % combine all data

end


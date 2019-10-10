function X_new = convertCategory(X)
%convertCategory
%   Convert vector of number categories X into a matrix of categories X_new
%   The columns correspond to what the categories are.
%   E.g. X = [1;3;2] --> X_new = [1,0,0;0,0,1;0,1,0]

[m,n] = size(X);            % m examples, n features
xmax = max(X,[],1);         % max of each feature
xmin = min(X,[],1);         % min of each feature
% n_cats = xmax-xmin + 1;     % num categories in each feature
n_cats = [6,6,2,2,2,10,6,6,4,2,26,20,2,2,2,2,21,2]; % n_cats in val. data

xshift = 1-xmin;
X_norm = X+xshift;          % shift categories with xmin=0 to start at 1

X_new = ones(m,1);          % add bias
for i=1:n
    I = eye(n_cats(i));
    X_add = I(X_norm(:,i),:);   % return matrix for feature categories
    X_new = [X_new X_add];      % add onto total matrix
end

end


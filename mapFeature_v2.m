function out = mapFeature_v2(X)
% MAPFEATURE_V2 Feature mapping function to polynomial features
%
%   MAPFEATURE_V2(X) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Input X must have columns as features (no bias) and rows as examples
%	This is only for polynomials of max degree = 2, don't add bias

[m,n] = size(X);
I = eye(n);

% Generate matrix of exponents of degree 1
E_1 = I;			% degree 1 features

% Generate matrix of exponents of degree 2
E_perm = zeros(n^2,n);
n_row_perm = 1;		% row counter
for i = 1:n			% permutation matrix
    for j = 1:n
        E_perm(n_row_perm,:) = I(i,:) + I(j,:);
		n_row_perm = n_row_perm+1;
    end
end
E_2 = unique(E_perm,'rows','stable');	% degree 2 features

% Assemble combined matrix of exponents
E_mat = [E_1;E_2];
n_mat = size(E_mat,1);

% New feature matrix
out = ones(m,1);	% bias column
for i = 1:n_mat
%     out(:,end+1) = prod(bsxfun(@power, X, E_mat(i,:)),2);	% bias
	out(:,i) = prod(bsxfun(@power, X, E_mat(i,:)),2);	% no bias
end

end
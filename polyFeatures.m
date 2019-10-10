function [X_poly] = polyFeatures(X,p)
% Takes data X matrix with n features/columns, then creates polynomial
% features of each column to the pth power.
% Don't add bias

[m,n_old] = size(X);
n_new = p*n_old;

X_pvec = bsxfun(@power,X(:),1:p);	% raise columns to exponents
X_poly = reshape(X_pvec,[m,n_new]);
% X_poly = [ones(m,1) X_poly];		% add bias
end
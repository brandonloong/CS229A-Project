function [X,y] = randomData(data,sub_size)
%randomData Select random a subset of data from original data
%   The data includes both features and y vals

% Get proper data set size
n_train = sub_size;
m_train = size(data,1);

% Randomly pick training rows for set i between [1,m_train]
rand_rows = round(rand(n_train,1)*(m_train-1)) + 1; % + 1 for zero ind    
rand_set = data(rand_rows,:);

X = rand_set(:,1:end-1);
y = rand_set(:,end);
end


clear all;
data = load('inmatesFirst1000.txt');
X = data(:,1:end-2);
y = data(:,end-1);
z = data(:,end);
clear data;

%{
[A mu sigma] = meanNormalize(X);
thetan = normalEqnLinear(X,y);
thetad = zeros(size(X,2),1); 
alpha = 0.003;
iters = 80000;
%}
%[theta J_hist] = gradientDescentLinear(X, y, thetad, alpha, iters);
%{
lambda = 0;
initial_theta = zeros(size(X, 2), 1);                
options = optimset('GradObj', 'on', 'MaxIter', 400);
%}
% [theta, J, exit_flag] = fminunc(@(t)(computeCostLogisticRegularized(t, X, z, lambda)), initial_theta, options);

input_layer_size  = 19;  
hidden_layer_size = 25;   
num_labels = 2;           

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Roll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];  

options = optimset('MaxIter', 500);
lambda = 0.05;                          
costFunction = @(p) computeCostNeuralNetwork(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, z+1, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
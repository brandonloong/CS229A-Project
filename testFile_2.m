clear
data = load('inmatesFirst1000.txt');
X = data(:,1:end-2);
y = data(:,end-1);
z = data(:,end);

%% Neural network on life sentences
lambda = 0;
i_theta = zeros(size(X, 2), 1);                
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta_fmin,J_fmin,exit_flag] = fminunc(@(t) (computeCostLogisticRegularized(t,X,z,lambda)), i_theta, options);

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
[nn_params,cost] = fmincg(costFunction,initial_nn_params,options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

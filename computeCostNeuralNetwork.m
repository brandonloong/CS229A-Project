  function [J grad] = computeCostNeuralNetwork(nn_params, input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   num_labels, (hidden_layer_size + 1));

  m = size(X, 1);                               % Setup some useful variables
  J = 0;                                        % You need to return the following variables correctly 
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));

  % ====================== YOUR CODE HERE ======================
  eye_matrix = eye(num_labels);
  y_matrix = eye_matrix(y,:);
  y_matrix = y_matrix';

  a1 = X;
  a1b = [ones(m, 1) X];
  z2 = a1b * Theta1';
  a2 = sigmoid(z2);
  a2b = [ones(m, 1) a2];
  z3 = a2b * Theta2';
  a3 = sigmoid(z3);

  h = a3;
  y_matrix = y_matrix';
  temp = (-y_matrix .* log(h)) - (1 - y_matrix) .* log(1 - h);
  d = sum(temp);
  err = (1/m) * sum(d);
  Theta1_sq = Theta1(:,2:end) .* Theta1(:,2:end);
  Theta2_sq = Theta2(:,2:end) .* Theta2(:,2:end);
  Theta1_sq_sum = sum(sum(Theta1_sq));
  Theta2_sq_sum = sum(sum(Theta2_sq));

  J = err + (lambda / (2 * m)) * (Theta1_sq_sum + Theta2_sq_sum);

  % PART 2: BACK PROPAGATION
  delta3 = (1 / m) * (a3 - y_matrix);
  Theta2_grad = delta3' * a2b;
  dlossda = delta3 * Theta2;
  dlossdz = dlossda .* a2b .* (1 - a2b);
  dlossdz = dlossdz(:,2:end);
  delta2 = dlossdz;
  Theta1_grad = delta2' * a1b;

  Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
  Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);

  grad = [Theta1_grad(:) ; Theta2_grad(:)];     % Unroll gradients

end
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------
% X = 5000 400 theta1 = 25  401 theta2 = 10  26
X = [ones(m,1) X];
A_1 = X;
Z_2 = X*Theta1'; % Z_2 = 5000 25
A_2 = sigmoid(Z_2); 
A_2 = [ones(m,1) A_2] ;
Z_3 = A_2*Theta2'; % Z_3 = 5000 * 10
A_3 = sigmoid(Z_3); 
Y = zeros(m,num_labels);
for i=1:m
  Y(i,y(i)) = 1;
end
%for i=1:m
%  for k=1:num_labels
%   J+= (y1(i,k)*log(A_3(i,k))+(1-y1(i,k))*log(1 - A_3(i,k)));
%  end
%end
%J = J*-1/m;
J = -(1/m) * sum(sum((Y.*log(A_3)) .+ (1.-Y).*log(1.-A_3)));
J = J + (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:, 2:end).^2))); %regularisation
% (2:end) since you don't want to regualrize the bias unit
% =========================================================================
delta_3 = A_3 - Y; % 5000 * 10
delta_2 = (delta_3*Theta2)(:,2:end).*sigmoidGradient(Z_2); % 5000*25
Theta2_grad = delta_3'*A_2./m; % delta_3 5000 10 A_2 5000 26 = 10*26
Theta1_grad = delta_2'*A_1./m; % 25 * 401

Theta2_grad(:,2:end) += lambda*Theta2(:,2:end)/m;
Theta1_grad(:,2:end) += lambda*Theta1(:,2:end)/m;
% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

function testModel()
%testModel Plot the model vs test set
%   Detailed explanation goes here

figure
plot(data_test(:,1:end-1)*theta-data_test(:,end))
xlabel('Model Y'); ylabel('Test Y');
title('Parity Plot');

end


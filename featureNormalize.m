function [X, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1.

feature_size = size(X, 2);
mu = zeros(1, feature_size);
sigma = zeros(1, feature_size);

for i = 1:feature_size
	mu(i) = mean(X(:,i));
	sigma(i) = std(X(:,i));
	X(:,i) = (X(:,i) - mu(i)) ./ sigma(i);
end

end

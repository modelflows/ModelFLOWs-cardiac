function [U sv] = svdtrunc(A, ns)
%SVDTRUNC Truncated SVD decomposition.
%	[U sv tol] = SVDTRUNC(A)
%	[U sv tol] = SVDTRUNC(A, tol)
%
%	A    - matrix
%	tol  - tolerance, only singular values > tol are kept (default: eps)
%
%	U    - truncated (left) singular vectors
%	sv   - list of truncated singular values
%	tol  - next singular value after truncation
%
%	eg. [U sv tol] = svdtrunc(rand(6,7), 0.5)
%
%	See also HOSVD, SVD.

if nargin == 1
	ns=length(sv);
end

[U S] = svd(A, 'econ');
sv=diag(S);

% if ns < length(sv)
% 	%tol = sv(ns+1);
% 	sv = sv(1:ns);
% 	U = U(:,1:ns);
% else
% 	%tol = 0;
% end

%Urest = U(:, ns+1:end);
%V = V(:, 1:ns);
%Vrest = V(:, ns+1:end);

function [TT, S, U, sv, n] = hosvd(T, n)
% HOSVD_STANDARD
%   Standard HOSVD (Tucker decomposition) for a multidimensional array.
%
% INPUTS:
%   T : multidimensional array (tensor)
%   n : vector of Tucker ranks, one per mode
%       n(i) = number of singular vectors to keep in mode i
%
% OUTPUTS:
%   TT : reconstructed tensor approximation (same size as T)
%   S  : core tensor such that approximately T ≈ tprod(S, U)
%   U  : cell array with factor matrices U{1},...,U{P}
%   sv : cell array with singular values (full, per mode)
%   n  : ranks actually used (same as input here)
%
% REQUIREMENTS:
%   - ndim_unfold(T, mode) must be available (your unfolding function)
%   - tprod must be available (tensor-times-matrix product in all modes)

    % Determine tensor size and number of modes
    M = size(T);
    P = length(M);

    if length(n) ~= P
        error('hosvd_standard: length(n) must equal the number of modes of T.');
    end

    % Allocate containers
    U  = cell(1,P);
    UT = cell(1,P);
    sv = cell(1,P);

    % SVD in each mode
    for mode = 1:P
        % Mode-n unfolding
        A = ndim_unfold(T, mode);     % size: M(mode) x (prod of other dims)

        % Economy SVD
        [Ui_full, S_mode, ~] = svd(A, 'econ');

        % Store all singular values for plotting / diagnostics
        sv{mode} = diag(S_mode);

        % Truncate to desired rank in this mode
        r = min(n(mode), size(Ui_full,2));
        Ui_trunc = Ui_full(:,1:r);

        % Optional: re-orthonormalize Ui_trunc (numerical stability)
        [Q, ~] = qr(Ui_trunc, 0);
        U{mode}  = Q(:,1:r);
        UT{mode} = U{mode}';
    end

    % Compute the core tensor: project T onto the subspaces U
    S = tprod(T, UT);   % S has size approximately n(1) x n(2) x ... x n(P)

    % Reconstruct the tensor from the core and factor matrices
    TT = tprod(S, U);
end
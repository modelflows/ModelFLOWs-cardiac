function [TT, S, U, sv, n] = hosvd_lc_multi3D(Tbig, Tred, n, IndexI, IndexJ, IndexK)
% HOSVD_LC_5D
%   Low-cost HOSVD for a 5D tensor using row subsampling (lcSVD idea)
%   in modes 2, 3 and 4.
%
% INPUTS:
%   Tbig   : full 5D tensor of size (ncomp, nx, ny, nz, nt)
%   Tred   : reduced 5D tensor = Tbig(:, IndexI, IndexJ, IndexK, :)
%   n      : Tucker ranks [n1, n2, n3, n4, n5]
%   IndexI : indices used in mode 2 (nx)   -> e.g. 1:2:nx
%   IndexJ : indices used in mode 3 (ny)   -> e.g. 1:4:ny
%   IndexK : indices used in mode 4 (nz)   -> e.g. 1:3:nz
%
% OUTPUTS:
%   TT  : reconstructed approximation of Tbig (same size as Tbig)
%   S   : core tensor
%   U   : cell array with factor matrices U{1},...,U{5}
%   sv  : singular values (per mode)
%   n   : ranks actually used (same as input here)
%
% NOTE:
%   - This function assumes that ndim_unfold(T, mode) and tprod are
%     already implemented (as in your previous HOSVD code).
%   - Modes:
%       mode 1 -> ncomp
%       mode 2 -> nx
%       mode 3 -> ny
%       mode 4 -> nz
%       mode 5 -> nt

    % Get full tensor size and number of modes
    Mbig = size(Tbig);
    P    = length(Mbig);   % should be 5

    if P ~= 5
        error('hosvd_lc_5d: Tbig must be a 5D tensor.');
    end

    % Basic consistency checks with Tred
    Mred = size(Tred);
    if Mred(1) ~= Mbig(1) || Mred(5) ~= Mbig(5)
        error('hosvd_lc_5d: Tred must share modes 1 and 5 with Tbig.');
    end
    if length(IndexI) ~= Mred(2) || length(IndexJ) ~= Mred(3) || length(IndexK) ~= Mred(4)
        error('hosvd_lc_5d: Tred must be Tbig(:,IndexI,IndexJ,IndexK,:).');
    end

    % Initialize containers
    U  = cell(1,P);
    UT = cell(1,P);
    sv = cell(1,P);

    % Row indices used for lcSVD in each mode
    % Empty means "no subsampling -> full SVD"
    indexRows = cell(1,P);
    indexRows{1} = [];        % mode 1: no subsampling
    indexRows{2} = IndexI;    % mode 2: use IndexI
    indexRows{3} = IndexJ;    % mode 3: use IndexJ
    indexRows{4} = IndexK;    % mode 4: use IndexK
    indexRows{5} = [];        % mode 5: no subsampling

    % Loop over modes
    for mode = 1:P

        % Unfold full tensor along current mode
        A_full = ndim_unfold(Tbig, mode);   % size: Mbig(mode) x (prod of others)

        r = n(mode);                        % desired rank in this mode

        if isempty(indexRows{mode})
            % ---------------------------------------------------------
            % No subsampling in this mode: standard HOSVD via SVD
            % ---------------------------------------------------------
            [Ui_full, S_i, ~] = svd(A_full, 'econ');

            % Truncate to rank r
            Ui_full = Ui_full(:,1:r);
            S_i     = S_i(1:r,1:r);

            % Optional: re-orthonormalize Ui_full
            [Q,~] = qr(Ui_full, 0);
            Ui = Q(:,1:r);

            U{mode}  = Ui;
            UT{mode} = U{mode}';
            sv{mode} = S_i;

        else
            % ---------------------------------------------------------
            % lcSVD in this mode: subsample rows as in LowCostSVD
            % ---------------------------------------------------------
            rows = indexRows{mode};

            % Reduced matrix: take selected rows from the full unfolding
            A_red = A_full(rows, :);        % size: (#rows) x (prod of others)

            % SVD in the reduced matrix (like Ared in LowCostSVD)
            [Ured, Sigma_red, Vred] = svd(A_red, 'econ');

            % Truncate to rank r
            Ured      = Ured(:,1:r);
            Vred      = Vred(:,1:r);
            Sigma_red = Sigma_red(1:r,1:r);

            % Re-orthonormalize Ured and Vred as in LowCostSVD
            [Q,R] = qr(Ured, 0);
            Ured  = Ured * inv(R(1:r,:));
            [Q,R] = qr(Vred, 0);
            Vred  = Vred * inv(R(1:r,:));

            % Sign correction using A_red (same spirit as your code)
            ss  = Ured' * A_red * Vred;
            ss1 = sign(diag(diag(ss)));
            Vred = Vred * ss1;

            % LIFT the left singular vectors to the full matrix:
            % This is analogous to U = A(:,indexJ)*Vred*inv(Sigma_red)
            % but here we use all columns, so directly:
            Ui_full = A_full * Vred * inv(Sigma_red);   % size: Mbig(mode) x r

            % Ensure orthonormality of Ui_full
            [Q,~] = qr(Ui_full, 0);
            Ui = Q(:,1:r);

            U{mode}  = Ui;
            UT{mode} = U{mode}';
            sv{mode} = Sigma_red;
        end
    end

    % Compute Tucker core using factor matrices U
    % S has (approximately) sizes (n(1), n(2), n(3), n(4), n(5))
    S  = tprod(Tbig, UT);

    % Reconstruct approximation of full tensor
    TT = tprod(S, U);
end
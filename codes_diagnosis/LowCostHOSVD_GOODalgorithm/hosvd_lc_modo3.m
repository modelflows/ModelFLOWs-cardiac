function [TT, S, U, sv, n] = hosvd_lc_modo3(Tbig, Tred, n, IndexK)
% HOSVD_LC_MODO3
%   HOSVD de bajo coste usando un tensor reducido sólo en el modo 3.
%
% INPUT:
%   Tbig   : tensor completo de tamaño (nx, ny, nz)
%   Tred   : tensor reducido = Tbig(:,:,IndexK)
%   n      : vector de rangos [n1 n2 n3]
%   IndexK : índices usados en la reducción del modo 3 (p.ej. 1:2:nz)
%
% OUTPUT:
%   TT  : aproximación del tensor completo Tbig (misma size que Tbig)
%   S   : núcleo Tucker
%   U   : cell{1,3} con las matrices de factores U{1},U{2},U{3}
%   sv  : valores singulares por modo (solo informativo)
%   n   : rangos usados (por si se modifican internamente)

    % Dimensiones
    Mbig = size(Tbig);      % [nx, ny, nz]
    P    = length(Mbig);    % debería ser 3

    if P ~= 3
        error('Esta versión está pensada para tensores 3D.');
    end

    % Comprobamos coherencia en modo 3
    Mred = size(Tred);      % [nx, ny, nz_red]
    if Mred(1) ~= Mbig(1) || Mred(2) ~= Mbig(2)
        error('Tred debe tener mismas dimensiones 1 y 2 que Tbig.');
    end
    if length(IndexK) ~= Mred(3)
        error('length(IndexK) debe coincidir con size(Tred,3).');
    end

    % Inicialización
    U  = cell(1,P);
    UT = cell(1,P);
    sv = cell(1,P);

    % --- Modo 1 y 2: HOSVD "normal" pero usando el tensor reducido ------
    % (el recorte está solo en la 3ª dimensión, así que A_red y A_full
    %  comparten la misma dimensión en filas para modos 1 y 2)
    for i = 1:2
        A_red = ndim_unfold(Tred, i);     % unfold del tensor reducido
        % Puedes usar svdtrunc o svd 'econ'. Aquí uso svdtrunc como en tu HOSVD:
        [Ui_red, svi_i] = svdtrunc(A_red, n(i));
        if n(i) < 2
            U{i} = [Ui_red(:,1), zeros(size(Ui_red(:,1)))];
        else
            U{i} = Ui_red(:,1:n(i));
        end
        UT{i} = U{i}';
        sv{i} = svi_i;
    end

    % --- Modo 3: lcSVD en el unfold del modo 3 ---------------------------
    % Afull: unfold completo en modo 3 (nz x (nx*ny))
    Afull = ndim_unfold(Tbig, 3);     % tamaño [nz, nx*ny]

    % A_red: filas seleccionadas según IndexK (lambda en k)
    A_red = Afull(IndexK, :);         % tamaño [nz_red, nx*ny]

    % SVD en la matriz reducida (igual que en LowCostSVD, pero 1D)
    [Ured, Sigma_red, Vred] = svd(A_red, 'econ');

    r = n(3);

    % Truncamos
    Ured      = Ured(:,1:r);
    Vred      = Vred(:,1:r);
    Sigma_red = Sigma_red(1:r,1:r);

    % Re-ortonormalizamos Ured y Vred como en tu LowCostSVD
    [Q,R] = qr(Ured,0);
    Ured  = Ured * inv(R(1:r,:));
    [Q,R] = qr(Vred,0);
    Vred  = Vred * inv(R(1:r,:));

    % Corrección de signos con A_red (análoga a tu código)
    ss  = Ured' * A_red * Vred;
    ss1 = sign(diag(diag(ss)));
    Vred = Vred * ss1;

    % Ahora levantamos U3 a partir de Afull (igual que U = A(:,indexJ)*...)
    % En tu LowCostSVD:
    %   U = A(:,indexJ)*Vred*inv(sigmared);
    % Aquí indexJ = todas las columnas, así que A(:,indexJ) = Afull:
    Ui_full = Afull * Vred * inv(Sigma_red);   % tamaño [nz x r]

    % Ortonormalizamos Ui_full para obtener U{3}
    [Q,~] = qr(Ui_full,0);
    Ui = Q(:,1:r);                 % [nz x r]

    U{3}  = Ui;
    UT{3} = U{3}';
    sv{3} = Sigma_red;

    % --- Construimos núcleo y reconstrucción sobre el tensor completo ----
    S  = tprod(Tbig, UT);   % núcleo sobre el tensor GRANDE
    TT = tprod(S, U);       % reconstrucción aproximada

end
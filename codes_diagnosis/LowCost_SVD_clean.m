% A=[1 2 3 4 5 6; 7 8 1 3 9 1; 1 1 5 3 9 1; 8 4 5 3 2 1; 9 4 6 8 1 3; 1 7 3 1 0 6; 2 6 2 7 1 4]
% indexI=[1:2:7];
% indexJ=[3:3:7];
% sel=2;

clc 
clear all
close all

%%%%
%% HAN YOU REPLACE LINE 12-23 by your original database
I=1000;
J=1000;

    for j=1:I
        for i=1:J
            %A(i,j)=rand-0.5;
            A(i,j)=ff(i/I,j/J)+1e-12*(rand);
           
        end
    end
indexI=[1:30:I];
indexJ=[1:25:J];
%%



% Create reduced matrix - 
%%%%%
%% HAN INCLUDE HERE YOU 3D MATRIX REDUCED AS I TOLD YOU, THIS WILL BE CALLED AS Ared 
%%%%
Ared=A(indexI,indexJ);
%%

% Calculate total number of points in Ared
'Number of points selected'
[length(indexI),length(indexJ)]
sel=30;  

% CALL lcSVD
[U,V,Sigma] = LowCostSVD(A,Ared, indexI,indexJ,sel);
Aapprox=U*Sigma*V';

% Calculate errors
DiferRed=A-Aapprox;
'error in SVD reduced'
norm(DiferRed(:),'inf')%/norm(A(:),'inf')

[UU,ssigma,VV]=svd(A,'econ');
DiferA=A-UU(:,1:sel)*ssigma(1:sel,1:sel)*VV(:,1:sel)';
'error in SVD'
norm(DiferA(:),'inf')%/norm(A(:),'inf')


%%
function [U,V,Sigma] = LowCostSVD(A,Ared, indexI,indexJ,sel)

%SVD in reduced matrix
[Ured,sigmared,Vred]=svd(Ared,'econ');
% Reduced modes in selected points
Ured=Ured(:,1:sel);
    % Re-orthonormalize Ured
    [Q,R]=qr(Ured);
    Ured=Ured*inv(R(1:sel,:));
Vred=Vred(:,1:sel);
    % Re-orthonornalize Vred
    [Q,R]=qr(Vred);
    Vred=Vred*inv(R(1:sel,:));
    % UPDATE SIGN Vred multiplying by the signs of Sigma_red:
    ss=Ured'*A(indexI,indexJ)*Vred;

    ss1=sign(diag(diag(ss)));
    Vred=Vred*ss1;
sigmared=sigmared(1:sel,1:sel);


% Calculate original SVD modes using the reduced modes
U=A(:,indexJ)*Vred*inv(sigmared);

V=A(indexI,:)'*Ured*inv(sigmared);

Sigma=sigmared;
 
end

function Ftest=ff(x,y)
 Ftest=2*x.^2*cos(2*(x*y+1)*(y-3))+sin(3*(x+2)*(y-1))-x.^3*y-x^2*y^2+sin(4*(x+2)*(y-pi));
end

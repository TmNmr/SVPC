%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Supplementary material for paper
% "Computational polyconvexification of isotropic functions"
% by T. Neumeier, M. Peter, D. Peterseim, D. Wiedemann
% University of Augsburg 
% 
% Copyright (C) 2023 T. Neumeier, M. Peter, D. Peterseim, D. Wiedemann 
% All Rights Reserved
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% script for demonstrating the performance of two variants of novel
% singular value polyconvexification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% examples 
exmpl = 'KSDo'; % (KSDo | dW2d | dW3d)
switch exmpl
    case 'KSDo' % Kohn Strang Dolzmann example (Section 4.1)
        % dimension
        d = 2; 
        % energy density
        W = @(F) (sqrt(sum(F.^2,2))>=sqrt(2)-1).*(1+sum(F.^2,2)) + ...
                 (sqrt(sum(F.^2,2))<sqrt(2)-1).*...
                 (2*sqrt(2).*sqrt(sum(F.^2,2)));
        % polyconvex hull
        mydet = @(F) F(:,1).*F(:,4)-F(:,2).*F(:,3);
        rho = @(F) sqrt(sum(F.^2,2)+2*abs(mydet(F)));
        Wpc = @(F) (rho(F)>1).*(1+sum(F.^2,2)) + ...
                   (rho(F)<=1).*2.*(rho(F)-abs(mydet(F)));
        % radius of bounding box B(r)
        r = 1.1; 
        % lattice size via refinement level
        lvl = 7;
        delta = 2*2^(-lvl)*r; 
        % evaluation point
        Fhat = [0.2 0.1;0.1 0.3];
    case 'dW2d' % double well 2D (Section 4.2)
        % dimension
        d = 2; 
        % energy density
        W = @(F) (sum(F.^2,2)-1).^2;
        % polyconvex hull
        Wpc = @(F) (sum(F.^2,2)>1).*W(F);
        % radius of bounding box B(r)
        r = 2; 
        % lattice size via refinement level
        lvl = 6;
        delta = 2*2^(-lvl)*r; 
        % evaluation point
        Fhat = [0.2 0.1;0.1 0.3];
    case 'dW3d' % double well 3D (Section 4.2)
        % dimension
        d = 3; 
        % energy density
        W = @(F) (sum(F.^2,2)-1).^2;
        % polyconvex hull
        Wpc = @(F) (sum(F.^2,2)>1).*W(F);
        % radius of bounding box B(r)
        r = 2; 
        % lattice size via refinement level
        lvl = 5;
        delta = 2*2^(-lvl)*r; 
        % evaluation point
        Fhat = eye(3)/3;
end

%% isotropic representation of energy density and exact polyconvex hull
if d == 2
    PHI = @(nu) W([nu(:,1) zeros(size(nu)) nu(:,2)]);
    PHIpc = @(nu) Wpc([nu(:,1) zeros(size(nu)) nu(:,2)]);
elseif d == 3
    PHI = @(nu) W([nu(:,1) zeros(size(nu)) nu(:,2) zeros(size(nu)) nu(:,3)]);
    PHIpc = @(nu) Wpc([nu(:,1) zeros(size(nu)) nu(:,2) zeros(size(nu)) nu(:,3)]);
end 

%% signed singular values
[~,S,~] = svd(Fhat);
nuhat = diag(S).';
if det(Fhat) < 0
    id = find(nuhat,1,'first');
    nuhat(id) = -nuhat(id);
end

%% polyconvexification
fprintf('Computing polyconvex hull of PHI =\n')
disp(W)
fprintf('at Fhat = \n')
disp(Fhat);
fprintf('in bounding box with radius r=%g, lattice size delta = %g ...\n\n',r,delta)
fprintf('W(Fhat) = %g\nWpc(Fhat) = %g (exact)\n',PHI(nuhat),PHIpc(nuhat));
if d<3 % use quickhull in 2d only
    % compute svpc hull using qhull
    c0 = cputime;
    WpchQhFhat = computePCHullQh(PHI,nuhat,r,delta);
    % display error
    eQh = abs(PHIpc(nuhat(:,1:d))-WpchQhFhat);
    fprintf('Wpc(Fhat) = %g (quickhull, error = %g, cputime = %g s) \n',...
             WpchQhFhat,max(eQh(eQh<Inf)),cputime - c0);
end
% compute svpc hull by linear programming
c0 = cputime;
WpchLpFhat = computePCHullLP(PHI,nuhat,r,delta);
% display error
eLp = abs(PHIpc(nuhat(:,1:d))-WpchLpFhat);
fprintf('Wpc(Fhat) = %g (linprog, error = %g, cputime = %gs)\n',...
         WpchLpFhat,max(eLp),cputime - c0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% supporting functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% singular value polyconvexification using quickhull
function [PHIpc] = computePCHullQh(PHI,nuhat,r,delta)
    %% detect dimension
    d = length(nuhat);
    %% Cartesian lattice of bounding box B(r) = B_max(0,r) ...
    switch d
        case 2
            [nu1,nu2] = ndgrid(linspace(-r,r,1+round(2*r/delta)));
            nu = [nu1(:),nu2(:)];
        case 3
            [nu1,nu2,nu3] = ndgrid(linspace(-r,r,1+round(2*r/delta)));
            nu = [nu1(:),nu2(:),nu3(:)];
    end
    %% lifting to minors' manifold
    X = minors(nu);
    Xhat = minors(nuhat);
    %% evaluate PHI in lattice points 
    h = PHI(X(:,1:d));
    %% point cloud p in k_d+1 dimensional space
    p = [X,h];
    %% compute convex hull of p using quickhull
    chull = convhulln(p,{'QJ'});
    %% determine lower convex hull via outer normals
    normals = computeNormals(chull,p);
    lchull = chull(normals(:,end)<-1e-10,:);
    %% evalute computed polyconvex hull at (remaining) points
    [id,xi] = findSimplex(lchull,X,Xhat);
    if id
        PHIpc = 0;
    else
        PHIpc = Inf;
    end
    for k = 1:size(lchull,2)
        PHIpc = PHIpc + p(lchull(id,k),end).*xi(k);
    end 
end % function computePCHullQh

%% singular value polyconvexification using linear programming
function [PHIpc] = computePCHullLP(PHI,nuhat,r,delta,tol)
    %% set default tolerance for interior point method
    if nargin<5
        tol = 1e-8;
    end
    %% detect dimension
    d = length(nuhat);
    %% Cartesian lattice of bounding box B(r) = B_max(0,r) ...
    switch d
        case 2
            [nu1,nu2] = ndgrid(linspace(-r,r,1+round(2*r/delta)));
            nu = [nu1(:),nu2(:)];
        case 3
            [nu1,nu2,nu3] = ndgrid(linspace(-r,r,1+round(2*r/delta)));
            nu = [nu1(:),nu2(:),nu3(:)];
    end
    %% lifting to minors' manifold
    X = minors(nu);
    Xhat = minors(nuhat);
    %% evaluate PHI in lattice points 
    h = PHI(X(:,1:d));
    %% approximate convex hull of [X PHInu] at xhat using linear programming
    Ndelta = size(X,1);
    options = optimoptions('linprog','Algorithm','Interior-Point-Legacy',...
                       'Display','none','MaxIterations',10000,...
                       'OptimalityTolerance',tol,'ConstraintTolerance',tol);
    [~,PHIpc,exitflag,output] = ...
        linprog(h,[],[],[X';ones(1,Ndelta)],[Xhat';1],zeros(Ndelta,1),[],options);
    if ~exitflag 
        warning('Solution of linprog not reliable!')
        display(output);
    end
end % function computePCHullLP

%% compute minors of vector nu
function m = minors(nu)
    d = size(nu, 2);
    switch d
        case 2
            m = [nu(:,1),nu(:,2),nu(:,1).*nu(:,2)];
        case 3
            m = [nu(:,1),nu(:,2),nu(:,3),nu(:,2).*nu(:,3),nu(:,3).*nu(:,1),...
                 nu(:,1).*nu(:,2),nu(:,1).*nu(:,2).*nu(:,3)];
    end
end % function minors

%% find element in simplicial mesh (t,p) that contains point x
function [id,b] = findSimplex(t,p,x)
    [nt,d] = size(t);
    idx = 1:size(x,1);
    id = zeros(size(x,1),1);
    b = zeros(size(x,1),size(t,2));
    for k = 1:nt
        bk = [p(t(k,:),:).';ones(1,d)]\[x(idx,:)';ones(1,length(idx))];
        idxk = find(abs(sum(abs(bk),1)-1) < 1e-10);
        if ~isempty(idxk)
            id(idx(idxk)) = k;
            b(idx(idxk),:) = bk(:,idxk).';
            idx = setdiff(idx,idx(idxk));
            if isempty(idx)
                break
            end
        end
    end
end % function findSimplex

%% compute outer normals of simplicial surface mesh
function normal = computeNormals(t,p)
    [nt,d] = size(t);
    intP = mean(p,1);
    outer = p(t(:,1),:) - ones(nt,1)*intP;         
    normal = zeros(size(t,1),size(p,2));
    tangents = p(t(:,2:end)',:) - p(repmat(t(:,1),1,d-1)',:);
    for k = 1:nt
        pk = tangents((k-1)*(d-1)+1:k*(d-1),:);
        [~,S,V] = svd(pk);
        if S(end,end-1) > 1e-12
            normal(k,:) = V(:,end);
            if normal(k,:)*outer(k,:)' < 0
                normal(k,:) = -normal(k,:);
            end
            normal(k,:) = normal(k,:)./norm(normal(k,:));
        end
    end
end % function computeNormals

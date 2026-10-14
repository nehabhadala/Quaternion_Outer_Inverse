
%% Based on paper "A complex structure-preserving algorithm for computing the
%% singular value decomposition of a quaternion matrix and its applications


%% This is code for Algorithm 3 given in above paper


%% complex representation technique to find SVD of a quaternion matrix


%% it works for any quternion matrix  A of size m*n, where m>n, m=n, or m<n
%% it works for all cases

%% Given matrix A=A0 + A1j where A0 and A1 are complex matrices

%% In this the input is A0,A1(Compoenets of quaternion matrix A=A0+A1j) and 
%% output are quaternion matrices Uq,Bq,Vq

%% such that A=Uq*Bq*Vq^H




function [Uq,Bq,Vq]=CSbidiagQ2(A0,A1)   % Complex Structure-Preserving Algorithm-SVD

[m,n]=size(A0);
mn=min(m,n);

% to save left and right transformation matrices 
U0=eye(m);
U1=zeros(m);
V0=eye(n);
V1=zeros(n);

if mn>=2   
    for s=1:mn-2
  %givens
        for t=s:m
            G1=CSGivensW(A0(t,s),A1(t,s));
            zz=G1*[A0(t,s:n);-conj(A1(t,s:n))];
            A0(t,s:n)= zz(1,:);
            A1(t,s:n)= -conj(zz(2,:));
            %
            zz=[U0(:,t) U1(:,t)]*G1';
            U0(:,t)=zz(:,1);
            U1(:,t)=zz(:,2);
        end
%         %householder
        [u,beta]=HouseW(A0(s:m,s),m+1-s);
        A0(s:m,s:n)=A0(s:m,s:n)-(beta*u)*(u'*A0(s:m,s:n));
        A1(s:m,s:n)=A1(s:m,s:n)-(beta*u)*(u'*A1(s:m,s:n));
        %

        U0(:,s:m)=U0(:,s:m)-(beta*U0(:,s:m)*u)*(u');
        U1(:,s:m)=U1(:,s:m)-(beta*U1(:,s:m)*u)*(u');
        
        %% V
%        givens 
       for t=s+1:n
            G1=CSGivensW(A0(s,t),A1(s,t));
            yy=[A0(s:m,t),A1(s:m,t)]*G1;
            A0(s:m,t)=yy(:,1);
            A1(s:m,t)=yy(:,2);
            %
            yy=[V0(:,t),V1(:,t)]*G1;
            V0(:,t)=yy(:,1);
            V1(:,t)=yy(:,2);
       end
%         householder  
        [v,beta]=HouseW(A0(s,s+1:n)',n-s);
        A0(s:m,s+1:n)=A0(s:m,s+1:n)-(A0(s:m,s+1:n)*v)*(beta*v');
        A1(s:m,s+1:n)=A1(s:m,s+1:n)-(A1(s:m,s+1:n)*v)*(beta*v');
        
        V0(:,s+1:n)=V0(:,s+1:n)-(V0(:,s+1:n)*v)*(beta*v');
        V1(:,s+1:n)=V1(:,s+1:n)-(V1(:,s+1:n)*v)*(beta*v');
     end
% s=mn-1, for the -2 row and column
    %
    s=mn-1;
%     U
%     givens
    for t= s:m
        G1=CSGivensW(A0(t,s),A1(t,s));
        zz=G1*[A0(t,s:n);-conj(A1(t,s:n));];
        A0(t,s:n)=zz(1,:);
        A1(t,s:n)= -conj(zz(2,:));
        %
        zz=[U0(:,t) U1(:,t)]*G1';
        U0(:,t)= zz(:,1);
        U1(:,t)= zz(:,2);
    end    
%householder
    [u,beta]=HouseW(A0(s:m,s),m+1-s);
    A0(s:m,s:n)=A0(s:m,s:n)-(beta*u)*(u'*A0(s:m,s:n));
    A1(s:m,s:n)=A1(s:m,s:n)-(beta*u)*(u'*A1(s:m,s:n));
    %
    U0(:,s:m)=U0(:,s:m)-(beta*U0(:,s:m)*u)*(u');
    U1(:,s:m)=U1(:,s:m)-(beta*U1(:,s:m)*u)*(u');
  %% V
    %givens
    for t=s+1:n
            G1=CSGivensW(A0(s,t),A1(s,t));
            yy=[A0(s:m,t),A1(s:m,t)]*G1;
            A0(s:m,t)=yy(:,1);
            A1(s:m,t)=yy(:,2);
        %
            yy=[V0(:,t),V1(:,t)]*G1;
            V0(:,t)=yy(:,1);
            V1(:,t)=yy(:,2);
    end

    if m<n
       % householder  
        [v,beta]=HouseW(A0(s,s+1:n)',n-s);
        A0(s:m,s+1:n)=A0(s:m,s+1:n)-(A0(s:m,s+1:n)*v)*(beta*v');
        A1(s:m,s+1:n)=A1(s:m,s+1:n)-(A1(s:m,s+1:n)*v)*(beta*v');
        
        V0(:,s+1:n)=V0(:,s+1:n)-(V0(:,s+1:n)*v)*(beta*v');
        V1(:,s+1:n)=V1(:,s+1:n)-(V1(:,s+1:n)*v)*(beta*v');
    end 

    %householder
    s=mn;
    % U
    %givens
    for t= s:m
        G1=CSGivensW(A0(t,s),A1(t,s));
        zz=G1*[A0(t,s:n);-conj(A1(t,s:n));];
        A0(t,s:n)=zz(1,:);
        A1(t,s:n)= -conj(zz(2,:));
        %
        zz=[U0(:,t) U1(:,t)]*G1';
        U0(:,t)= zz(:,1);
        U1(:,t)= zz(:,2);
    end
%      %householder
     if m>n
        [u,beta]=HouseW(A0(s:m,s),m+1-s);
        A0(s:m,n)=A0(s:m,n)-(beta*u)*(u'*A0(s:m,n));
        A1(s:m,n)=A1(s:m,n)-(beta*u)*(u'*A1(s:m,n));
        %
        %
        U0(:,s:m)=U0(:,s:m)-(beta*U0(:,s:m)*u)*(u');
        U1(:,s:m)=U1(:,s:m)-(beta*U1(:,s:m)*u)*(u');
     end 

     if m<n
        %givens
       for t=s+1:n
            G1=CSGivensW(A0(s,t),A1(s,t));
            yy=[A0(s:m,t),A1(s:m,t)]*G1;
            A0(s:m,t)=yy(:,1);
            A1(s:m,t)=yy(:,2);
            %
            yy=[V0(:,t),V1(:,t)]*G1;
            V0(:,t)=yy(:,1);
            V1(:,t)=yy(:,2);
       end

      % householder  
        [v,beta]=HouseW(A0(s,s+1:n)',n-s);
        A0(s:m,s+1:n)=A0(s:m,s+1:n)-(A0(s:m,s+1:n)*v)*(beta*v');
        A1(s:m,s+1:n)=A1(s:m,s+1:n)-(A1(s:m,s+1:n)*v)*(beta*v');
        
        V0(:,s+1:n)=V0(:,s+1:n)-(V0(:,s+1:n)*v)*(beta*v');
        V1(:,s+1:n)=V1(:,s+1:n)-(V1(:,s+1:n)*v)*(beta*v');
     end 


end

     
     
     
U=[U0, U1];
V=[V0,V1];
B=A0;
[u,B,v]=svd(B);
U=U*blkdiag(u,u);
V=V*blkdiag(v,v);

M1=U(:,1 : m);
M2=U(:,(m+1) : 2*m);

N1=V(:,1 : n);
N2=V(:,(n+1) : 2*n);

Uq=quaternion(real(M1),imag(M1),real(M2),imag(M2));
Vq=quaternion(real(N1),imag(N1),real(N2),imag(N2));
Bq=B;
end



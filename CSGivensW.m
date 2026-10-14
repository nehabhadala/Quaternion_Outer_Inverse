function Gw=CSGivensW(g0,g1)
    g0 = conj(g0); g1 = -g1;
    sq=sqrt(g0*conj(g0)+g1*conj(g1));
    tol=1e-14;
    if sq<=tol
        Gw=eye(2);
    else
        Gw=[ g0 g1; -conj(g1) conj(g0)]/sq; 
    end
end

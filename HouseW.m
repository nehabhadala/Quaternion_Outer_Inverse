function [v, fi] = HouseW(x, n)
    v = x;
    l = norm(x);
    if v(1) >= 0
        v(1) = v(1) + l;
        fi = 1 / (v(1) * l);
    else
        v(1) = v(1) - l;
        fi = -1 / (v(1) * l); 
    end
end

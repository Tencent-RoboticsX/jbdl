function v = LinearSpline(vstart, vend, ratio)
%LinearSpline - Description
%
% Syntax: v = LinearSpline(tstart, tend, ratio)
%
% Long description
    v = ratio * vend + (1.0 - ratio) * vstart;
end
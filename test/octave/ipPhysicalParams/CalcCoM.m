function c = CalcCoM(CoM, Mass)
%CalcCoM - Calculate center of mass
%
% Syntax: c = CalcCoM(CoM, Mass)
%
% CoM: CoM matrix, a combination of column vectors  
% Mass: mass vector

[hcom, lcom] = size(CoM);
[hmass, lmass] = size(Mass);

if hcom~=3 
    error('Error: CoM array shoule be a combination of column vectors!');
    c = zeros(3,1);
elseif lmass~=1
    error('Error: Mass array shoule be !');
    c = zeros(3,1);
elseif lcom~=hmass
    error('Error: The dimension of CoM and Mass array is inconsistent!');
    c = zeros(3,1);
else
    mc = zeros(3,1);
    m = 0;
    for i=1:hmass
        mc = mc + CoM(:, i)*Mass(i);
        m = m + Mass(i);
    end

    c = mc / m;
end
    
end
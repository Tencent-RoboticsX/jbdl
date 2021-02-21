function I = CalcInertia_ParallelAxisTheorem(Icom, Pcom, mass)
%CalcInertia_ParalleAxisTheorem - Calculate inertia from CoM frame to a new frame
%
% Syntax: I = CalcInertia_ParalleAxisTheorem(Icom, Pcom)
%
% Icom: inertia reletive to CoM
% Pcom: CoM position in new frame
% mass: mass
% I: inertia reletive to new frame
    I = zeros(3);

    x = Pcom(1);
    y = Pcom(2);
    z = Pcom(3);
    I(1,1) = Icom(1,1) + mass * (y*y + z*z);
    I(2,2) = Icom(2,2) + mass * (x*x + z*z);
    I(3,3) = Icom(3,3) + mass * (x*x + y*y);

    I(1,2) = Icom(1,2) - mass * x * y;
    I(1,3) = Icom(1,3) - mass * x * z;
    I(2,3) = Icom(2,3) - mass * y * z;
    Icom(2,1) = I(1,2);
    Icom(3,1) = I(1,3);
    Icom(3,2) = I(2,3);
    
end
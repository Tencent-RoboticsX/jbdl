function lxyz = CalcInertiaCubiod(Inertia, mass)
%CalcInertiaCubiod - Calculate a box that the sizes and orientations of the boxes 
% correspond to unit-mass boxes with the same inertial behavior as 
% their corresponding links. 
%
% Syntax: lxyz = CalcInertiaCubiod(Inertia)
%
% Inertia: [Ixx, Iyy, Izz]'
% mass : mass

A = [0, 1, 1; 1, 0, 1; 1, 1, 0];
B = 1/12*mass*Inertia;
C = inv(A)*B;

lxyz = zeros(3,1);
for i=1:3
    lxyz(i) = sqrt(C(i));
end
    
end
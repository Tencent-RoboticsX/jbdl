function I = ipCalcCylinderMainInertial(r, h, mass)
%% Calculate the main inertial of cylinder
% r: radius 
% h: height
% Note: the height is along y-axis
    
    Iyy = 1/2*mass*r*r;
    Ixx = 1/12*mass*(3*r*r+h*h);
    Izz = 1/12*mass*(3*r*r+h*h);

    I = [Ixx, Iyy, Izz]';

end
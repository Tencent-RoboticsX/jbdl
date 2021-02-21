function I = ipCalcMainInertial(box, mass)
    % function: 计算长方体的主惯量矩
    % input:  box: array 3x1, length, width, height
    %        mass: whole mass
    % output: I: array 3x1, Ixx, Iyy, Izz
    % note:  x --> length, y --> width, z --> height
    
    l = box(1); 
    w = box(2);
    h = box(3);
    
    Ixx = 1/12*mass*(w^2+h^2);
    Iyy = 1/12*mass*(l^2+h^2);
    Izz = 1/12*mass*(l^2+w^2);
    
    I = [Ixx, Iyy, Izz]';

end

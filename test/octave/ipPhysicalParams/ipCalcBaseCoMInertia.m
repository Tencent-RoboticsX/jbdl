function [m, c, I] = ipCalcBaseCoMInertia()
%ipCalcBaseCoMInertia - Calculate CoM and inertia of planar link's base
%
% Syntax: [m, c, I] = ipCalcBaseCoMInertia()
%
% Long description

    global ip;
    % Calculate mass
    m = ip.mass_base + ip.mass_abad_fr + ip.mass_abad_fl + ip.mass_abad_br + ip.mass_abad_bl;

    % Calculate CoM reletive to base frame
    com_abad_fr_in_base = ip.com_abad_fr + ip.r_base_abad_fr; 
    com_abad_fl_in_base = ip.com_abad_fl + ip.r_base_abad_fl; 
    com_abad_br_in_base = ip.com_abad_br + ip.r_base_abad_br; 
    com_abad_bl_in_base = ip.com_abad_bl + ip.r_base_abad_bl; 

    com = [ip.com_base, com_abad_fr_in_base, com_abad_fl_in_base, com_abad_br_in_base, com_abad_bl_in_base];
    mass = [ip.mass_base, ip.mass_abad_fr, ip.mass_abad_fl, ip.mass_abad_br, ip.mass_abad_bl]';
    
    c = CalcCoM(com, mass);

    % Calculate inertia reletive to base's CoM frame
    com_abad_fr_in_basecom = com_abad_fr_in_base - c;
    com_abad_fl_in_basecom = com_abad_fl_in_base - c;
    com_abad_br_in_basecom = com_abad_br_in_base - c;
    com_abad_bl_in_basecom = com_abad_bl_in_base - c;

    inertia_abad_fr_in_basecom = CalcInertia_ParallelAxisTheorem(ip.inertia_abad_fr, com_abad_fr_in_basecom, ip.mass_abad_fr);
    inertia_abad_fl_in_basecom = CalcInertia_ParallelAxisTheorem(ip.inertia_abad_fl, com_abad_fl_in_basecom, ip.mass_abad_fl);
    inertia_abad_br_in_basecom = CalcInertia_ParallelAxisTheorem(ip.inertia_abad_br, com_abad_br_in_basecom, ip.mass_abad_br);
    inertia_abad_bl_in_basecom = CalcInertia_ParallelAxisTheorem(ip.inertia_abad_bl, com_abad_bl_in_basecom, ip.mass_abad_bl);

    I = ip.inertia_base + inertia_abad_fr_in_basecom + inertia_abad_fl_in_basecom + inertia_abad_br_in_basecom + inertia_abad_bl_in_basecom;

end
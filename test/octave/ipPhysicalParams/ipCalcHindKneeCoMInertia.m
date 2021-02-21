function [m, c, I] = CalcHindKneeCoMInertia()
%CalcHindKneeCoMInertia - Description
%
% Syntax: [m, c, I] = CalcHindKneeCoMInertia()
%
% Long description
    global ip;
    % Calculate mass
    m = ip.mass_knee_br + ip.mass_knee_bl + ip.mass_wheel_br + ip.mass_wheel_bl;

    % Calculate CoM reletive to hind knee frame
    com_knee_br_in_base = ip.com_knee_br + ip.r_hhip_knee_br + ip.r_abad_hhip_br + ip.r_base_abad_br;
    com_knee_bl_in_base = ip.com_knee_bl + ip.r_hhip_knee_bl + ip.r_abad_hhip_bl + ip.r_base_abad_bl;
    com_wheel_br_in_base = ip.com_wheel_br + ip.r_knee_wheel_br + ip.r_hhip_knee_br + ip.r_abad_hhip_br + ip.r_base_abad_br;
    com_wheel_bl_in_base = ip.com_wheel_bl + ip.r_knee_wheel_bl + ip.r_hhip_knee_bl + ip.r_abad_hhip_bl + ip.r_base_abad_bl;

    com_hindknee_in_base = [-ip.length_body*0.5, 0., -ip.length_hhip]';

    com_knee_br_in_hindknee = com_knee_br_in_base - com_hindknee_in_base;
    com_knee_bl_in_hindknee = com_knee_bl_in_base - com_hindknee_in_base;
    com_wheel_br_in_hindknee = com_wheel_br_in_base - com_hindknee_in_base;
    com_wheel_bl_in_hindknee = com_wheel_bl_in_base - com_hindknee_in_base;

    com = [com_knee_br_in_hindknee, com_knee_bl_in_hindknee, com_wheel_br_in_hindknee, com_wheel_bl_in_hindknee];
    mass = [ip.mass_knee_br, ip.mass_knee_bl, ip.mass_wheel_br, ip.mass_wheel_bl]';

    c = CalcCoM(com, mass);

    % Calculate inertia reletive to hind knee's CoM frame
    com_knee_br_in_hindkneecom = com_knee_br_in_hindknee - c;
    com_knee_bl_in_hindkneecom = com_knee_bl_in_hindknee - c;
    com_wheel_br_in_hindkneecom = com_wheel_br_in_hindknee - c;
    com_wheel_bl_in_hindkneecom = com_wheel_bl_in_hindknee - c;

    inertia_knee_br_in_hindkneecom = CalcInertia_ParallelAxisTheorem(ip.inertia_knee_br, com_knee_br_in_hindkneecom, ip.mass_knee_br);
    inertia_knee_bl_in_hindkneecom = CalcInertia_ParallelAxisTheorem(ip.inertia_knee_bl, com_knee_bl_in_hindkneecom, ip.mass_knee_bl);
    inertia_wheel_br_in_hindkneecom = CalcInertia_ParallelAxisTheorem(ip.inertia_wheel_br, com_wheel_br_in_hindkneecom, ip.mass_wheel_br);
    inertia_wheel_bl_in_hindkneecom = CalcInertia_ParallelAxisTheorem(ip.inertia_wheel_bl, com_wheel_bl_in_hindkneecom, ip.mass_wheel_bl);

    I = inertia_knee_br_in_hindkneecom + inertia_knee_bl_in_hindkneecom + inertia_wheel_br_in_hindkneecom + inertia_wheel_bl_in_hindkneecom;

end
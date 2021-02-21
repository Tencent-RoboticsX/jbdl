function [m, c, I] = CalcForeKneeCoMInertia()
%CalcForeKneeCoMInertia - Description
%
% Syntax: [m, c, I] = CalcForeKneeCoMInertia()
%
% Long description
    global ip;
    % Calculate mass
    m = ip.mass_knee_fr + ip.mass_knee_fl + ip.mass_wheel_fr + ip.mass_wheel_fl;

    % Calculate CoM reletive to fore knee frame
    com_knee_fr_in_base = ip.com_knee_fr + ip.r_hhip_knee_fr + ip.r_abad_hhip_fr + ip.r_base_abad_fr;
    com_knee_fl_in_base = ip.com_knee_fl + ip.r_hhip_knee_fl + ip.r_abad_hhip_fl + ip.r_base_abad_fl;
    com_wheel_fr_in_base = ip.com_wheel_fr + ip.r_knee_wheel_fr + ip.r_hhip_knee_fr + ip.r_abad_hhip_fr + ip.r_base_abad_fr;
    com_wheel_fl_in_base = ip.com_wheel_fl + ip.r_knee_wheel_fl + ip.r_hhip_knee_fl + ip.r_abad_hhip_fl + ip.r_base_abad_fl;

    com_foreknee_in_base = [ip.length_body*0.5, 0., -ip.length_hhip]';

    com_knee_fr_in_foreknee = com_knee_fr_in_base - com_foreknee_in_base;
    com_knee_fl_in_foreknee = com_knee_fl_in_base - com_foreknee_in_base;
    com_wheel_fr_in_foreknee = com_wheel_fr_in_base - com_foreknee_in_base;
    com_wheel_fl_in_foreknee = com_wheel_fl_in_base - com_foreknee_in_base;

    com = [com_knee_fr_in_foreknee, com_knee_fl_in_foreknee, com_wheel_fr_in_foreknee, com_wheel_fl_in_foreknee];
    mass = [ip.mass_knee_fr, ip.mass_knee_fl, ip.mass_wheel_fr, ip.mass_wheel_fl]';

    c = CalcCoM(com, mass);

    % Calculate inertia reletive to fore knee's CoM frame
    com_knee_fr_in_forekneecom = com_knee_fr_in_foreknee - c;
    com_knee_fl_in_forekneecom = com_knee_fl_in_foreknee - c;
    com_wheel_fr_in_forekneecom = com_wheel_fr_in_foreknee - c;
    com_wheel_fl_in_forekneecom = com_wheel_fl_in_foreknee - c;

    inertia_knee_fr_in_forekneecom = CalcInertia_ParallelAxisTheorem(ip.inertia_knee_fr, com_knee_fr_in_forekneecom, ip.mass_knee_fr);
    inertia_knee_fl_in_forekneecom = CalcInertia_ParallelAxisTheorem(ip.inertia_knee_fl, com_knee_fl_in_forekneecom, ip.mass_knee_fl);
    inertia_wheel_fr_in_forekneecom = CalcInertia_ParallelAxisTheorem(ip.inertia_wheel_fr, com_wheel_fr_in_forekneecom, ip.mass_wheel_fr);
    inertia_wheel_fl_in_forekneecom = CalcInertia_ParallelAxisTheorem(ip.inertia_wheel_fl, com_wheel_fl_in_forekneecom, ip.mass_wheel_fl);

    I = inertia_knee_fr_in_forekneecom + inertia_knee_fl_in_forekneecom + inertia_wheel_fr_in_forekneecom + inertia_wheel_fl_in_forekneecom;

end
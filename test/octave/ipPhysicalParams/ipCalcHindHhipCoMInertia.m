function [m, c, I] = CalcForeHhipCoMInertia()
%CalcForeHhipCoMInertia - Description
%
% Syntax: [m, c, I] = CalcForeHhipCoMInertia()
%
% Long description
    global ip;
    % Calculate mass
    m = ip.mass_hhip_br + ip.mass_hhip_bl;

    % Calculate CoM reletive to hind hip frame
    com_hhip_br_in_base = ip.com_hhip_br + ip.r_abad_hhip_br + ip.r_base_abad_br;
    com_hhip_bl_in_base = ip.com_hhip_bl + ip.r_abad_hhip_bl + ip.r_base_abad_bl;

    com_hindhip_in_base = [-ip.length_body*0.5, 0., 0.]';

    com_hhip_br_in_hindhip = com_hhip_br_in_base - com_hindhip_in_base;
    com_hhip_bl_in_hindhip = com_hhip_bl_in_base - com_hindhip_in_base;

    com = [com_hhip_br_in_hindhip, com_hhip_bl_in_hindhip];
    mass = [ip.mass_hhip_fr, ip.mass_hhip_fl]';

    c = CalcCoM(com, mass);

    % Calculate inertia reletive to hind hip's CoM frame
    com_hhip_br_in_hindhipcom = com_hhip_br_in_hindhip - c;
    com_hhip_bl_in_hindhipcom = com_hhip_bl_in_hindhip - c;

    inertia_hhip_br_in_hindhipcom = CalcInertia_ParallelAxisTheorem(ip.inertia_hhip_br, com_hhip_br_in_hindhipcom, ip.mass_hhip_fr);
    inertia_hhip_bl_in_hindhipcom = CalcInertia_ParallelAxisTheorem(ip.inertia_hhip_bl, com_hhip_bl_in_hindhipcom, ip.mass_hhip_fl);

    I = inertia_hhip_br_in_hindhipcom + inertia_hhip_bl_in_hindhipcom;

end
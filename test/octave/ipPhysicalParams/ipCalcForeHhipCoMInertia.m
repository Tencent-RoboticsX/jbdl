function [m, c, I] = CalcForeHhipCoMInertia()
%CalcForeHhipCoMInertia - Description
%
% Syntax: [m, c, I] = CalcForeHhipCoMInertia()
%
% Long description
    global ip;
    % Calculate mass
    m = ip.mass_hhip_fr + ip.mass_hhip_fl;

    % Calculate CoM reletive to fore hip frame
    com_hhip_fr_in_base = ip.com_hhip_fr + ip.r_abad_hhip_fr + ip.r_base_abad_fr;
    com_hhip_fl_in_base = ip.com_hhip_fl + ip.r_abad_hhip_fl + ip.r_base_abad_fl;

    com_forehip_in_base = [ip.length_body*0.5, 0., 0.]';

    com_hhip_fr_in_forehip = com_hhip_fr_in_base - com_forehip_in_base;
    com_hhip_fl_in_forehip = com_hhip_fl_in_base - com_forehip_in_base;

    com = [com_hhip_fr_in_forehip, com_hhip_fl_in_forehip];
    mass = [ip.mass_hhip_fr, ip.mass_hhip_fl]';

    c = CalcCoM(com, mass);

    % Calculate inertia reletive to fore hip's CoM frame
    com_hhip_fr_in_forehipcom = com_hhip_fr_in_forehip - c;
    com_hhip_fl_in_forehipcom = com_hhip_fl_in_forehip - c;

    inertia_hhip_fr_in_forehipcom = CalcInertia_ParallelAxisTheorem(ip.inertia_hhip_fr, com_hhip_fr_in_forehipcom, ip.mass_hhip_fr);
    inertia_hhip_fl_in_forehipcom = CalcInertia_ParallelAxisTheorem(ip.inertia_hhip_fl, com_hhip_fl_in_forehipcom, ip.mass_hhip_fl);

    I = inertia_hhip_fr_in_forehipcom + inertia_hhip_fl_in_forehipcom;

end
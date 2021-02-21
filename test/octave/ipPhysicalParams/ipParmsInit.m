function init_ip = ipParmsInit(nx, nu, ny, noc)
    global ip;

    % length
    ip.length_body = 0.392; % length_body + length_abad * 2.0
    ip.width_body = 0.120;
    ip.height_body = 0.10;
    ip.length_abad = 0.102;
    ip.length_hhip = 0.2115;
    ip.length_knee = 0.265;
    ip.radius_wheel_fr = 0.028;
    ip.radius_wheel_fl = 0.028;
    ip.radius_wheel_br = 0.036;
    ip.radius_wheel_bl = 0.036;
    ip.bias_wheel_br = 0.0558;
    ip.bias_wheel_bl =0.0558;

    % mass
    ip.mass_floatbase = 0.;
    ip.mass_base =  4.567049;
    ip.mass_abad_fr = 0.593025;
    ip.mass_abad_fl = 0.593025;
    ip.mass_hhip_fr = 1.061860;
    ip.mass_hhip_fl = 1.061860;
    ip.mass_knee_fr = 0.167615;
    ip.mass_knee_fl = 0.167615;
    ip.mass_abad_br = 0.593025;
    ip.mass_abad_bl = 0.593025;
    ip.mass_hhip_br = 1.051039;
    ip.mass_hhip_bl = 1.051039;
    ip.mass_knee_br = 0.283846;
    ip.mass_knee_bl = 0.283846;
    ip.mass_wheel_fr = 0.0629346;
    ip.mass_wheel_fl = 0.0629346;
    ip.mass_wheel_br = 0.12;
    ip.mass_wheel_bl = 0.12;

    % CoM position reletive to joint frame
    ip.com_base = [0.030686119093162, -0.000329627701830426, 0.00547786510534019]';
    ip.com_abad_fr = [-0.00245822551833624, 0.00119871971453624, -3.4465266880862E-06]';
    ip.com_hhip_fr = [6.94812145732304E-05, 0.0270939953144388, -0.0242692939783491]';
    ip.com_knee_fr = [0.00195996669829623, -2.98942005325931E-05, -0.131269459282808]';
    ip.com_abad_fl = [-0.00245822551833624, -0.00119871971453624, -3.4465266880862E-06]';
    ip.com_hhip_fl = [6.94812145732304E-05, -0.0270939953144388, -0.0242692939783491]';
    ip.com_knee_fl = [0.00195996669829623, 2.98942005325931E-05, -0.131269459282808]';
    ip.com_abad_br = [0.00245822551832164, 0.00119871971454771, 3.44618013202458E-06]';
    ip.com_hhip_br = [0.000200180750027534, 0.0276087447540674, -0.0211018649999141]';
    ip.com_knee_br = [0.00146519520473593, 7.17111183709679E-05, -0.0958381396031051]';
    ip.com_abad_bl = [0.00245822551832164, -0.00119871971454771, 3.44618013202458E-06]';
    ip.com_hhip_bl = [0.000200180750027534, -0.0276087447540674, -0.0211018649999141]';
    ip.com_knee_bl = [0.00146519520473593, -7.17111183709679E-05, -0.0958381396031051]';
    ip.com_wheel_fr = [0., 0.012000, 0.]';
    ip.com_wheel_fl = [0., -0.012000, 0.]'; 
    ip.com_wheel_br = [0., 0.01000, 0.]';
    ip.com_wheel_bl = [0., -0.01000, 0.]'; 
    
    % Link inretial reletive to joint frame
    ip.inertia_base = [0.01804806, 0.00000867, 0.00149180; 0.00000867, 0.06353002, -0.00002865; 0.00149180, -0.00002865, 0.07571430;];
    ip.inertia_abad_fr = [0.00034999, -0.00000504, -0.00000019; -0.00000504, 0.00060739, 0.00000003; -0.00000019, 0.00000003, 0.00039751;];
    ip.inertia_hhip_fr = [0.00544534, -0.00000285, -0.00000170; -0.00000285, 0.00534218, -0.00082154; -0.00000170, -0.00082154, 0.00122467;];
    ip.inertia_knee_fr = [0.00150924, 0.00000008, -0.00001734; 0.00000008, 0.00152901, -0.00000045; -0.00001734, -0.00000045, 0.00003686;];
    ip.inertia_abad_fl = [0.00034999, -0.00000504, -0.00000019; -0.00000504, 0.00060739, 0.00000003; -0.00000019, 0.00000003, 0.00039751;];
    ip.inertia_hhip_fl = [0.00544534, 0.00000285, -0.00000170; 0.00000285, 0.00534218, 0.00082154; -0.00000170, 0.00082154, 0.00122467;];
    ip.inertia_knee_fl = [0.00150924, -0.00000008, -0.00001734; -0.00000008, 0.00152901, 0.00000045; -0.00001734, 0.00000045, 0.00003686;];
    ip.inertia_abad_br = [0.00034999, -0.00000504, -0.00000019; -0.00000504, 0.00060739, 0.00000003; -0.00000019, 0.00000003, 0.00039751;];
    ip.inertia_hhip_br = [0.00478794, 0.00000412, 0.00002708; 0.00000412, 0.00474336, -0.00063896; 0.00002708, -0.00063896, 0.00117668;];
    ip.inertia_knee_br = [0.00222705, -0.00000013, 0.00000387; -0.00000013, 0.00225834, -0.00000199; 0.00000387, -0.00000199, 0.00006895;];
    ip.inertia_abad_bl = [0.00034999, -0.00000504, -0.00000019; -0.00000504, 0.00060739, 0.00000003; -0.00000019, 0.00000003, 0.00039751;];
    ip.inertia_hhip_bl = [0.00478794, -0.00000412, 0.00002708; -0.00000412, 0.00474336, 0.00063896; 0.00002708, 0.00063896, 0.00117668;];
    ip.inertia_knee_bl = [0.00222705, -0.00000013, 0.00000387; -0.00000013, 0.00225834, -0.00000199; 0.00000387, -0.00000199, 0.00006895;];
    ip.inertia_wheel_fr = [0.00001749, 0., 0.; 0., 0.00002918, 0.; 0., 0., 0.00001749;];
    ip.inertia_wheel_fl = [0.00001749, 0., 0.; 0., 0.00002918, 0.; 0., 0., 0.00001749;];
    ip.inertia_wheel_br = [0.000025, 0., 0.; 0., 0.000040, 0.; 0., 0., 0.000025;];
    ip.inertia_wheel_bl = [0.000025, 0., 0.; 0., 0.000040, 0.; 0., 0., 0.000025;];

    % reletive position in plannar frame  
    ip.r_base_abad_fr = [0.5 * ip.length_body, -0.5 * ip.width_body, 0.]';
    ip.r_abad_hhip_fr = [0., -ip.length_abad, 0.]';
    ip.r_hhip_knee_fr = [0., 0., -ip.length_hhip]';
    ip.r_knee_endp_fr = [0., 0., -ip.length_knee]';
    ip.r_knee_wheel_fr = [0., -ip.bias_wheel_br, 0.]';
    ip.r_base_abad_fl = [0.5 * ip.length_body, 0.5 * ip.width_body, 0.]';
    ip.r_abad_hhip_fl = [0., ip.length_abad, 0.]'; 
    ip.r_hhip_knee_fl = [0., 0., -ip.length_hhip]';
    ip.r_knee_endp_fl = [0., 0., -ip.length_knee]';
    ip.r_knee_wheel_fl = [0., ip.bias_wheel_bl, 0.]';
    ip.r_base_abad_br = [-0.5 * ip.length_body, -0.5 * ip.width_body, 0.]';
    ip.r_abad_hhip_br = [0., -ip.length_abad, 0.]'; 
    ip.r_hhip_knee_br = [0., 0., -ip.length_hhip]';
    ip.r_knee_endp_br = [0., 0., -ip.length_knee]';
    ip.r_knee_wheel_br = [0., -ip.bias_wheel_br, 0.]';
    ip.r_base_abad_bl = [-0.5 * ip.length_body, 0.5 * ip.width_body, 0.]';
    ip.r_abad_hhip_bl = [0., ip.length_abad, 0.]'; 
    ip.r_hhip_knee_bl = [0., 0., -ip.length_hhip]';
    ip.r_knee_endp_bl = [0., 0., -ip.length_knee]';
    ip.r_knee_wheel_bl = [0., ip.bias_wheel_bl, 0.]';

    ip.axis_0 = zeros(3,1);
    ip.axis_x = [1.0, 0.0, 0.0]';
    ip.axis_y = [0.0, 1.0, 0.0]';
    ip.axis_z = [0.0, 0.0, 1.0]';

    % Calculate planar link CoM and Inertia
    [ip.Massbase, ip.CoMbase, ip.Inertiabase] = ipCalcBaseCoMInertia();
    [ip.MassForeHhip, ip.CoMForeHhip, ip.InertiaForeHhip] = ipCalcForeHhipCoMInertia();
    [ip.MassHindHhip, ip.CoMHindHhip, ip.InertiaHindHhip] = ipCalcHindHhipCoMInertia();
    [ip.MassForeKnee, ip.CoMForeKnee, ip.InertiaForeKnee] = ipCalcForeKneeCoMInertia();
    [ip.MassHindKnee, ip.CoMHindKnee, ip.InertiaHindKnee] = ipCalcHindKneeCoMInertia();

    ip.thigh_l = 0.05;
    ip.thigh_w = 0.03;
    ip.shank_l = 0.04;
    ip.shank_w = 0.02;
    ip.thigh_h = 0.2115;
    ip.shank_h = 0.265;
    ip.fore_wheel_thick = 0.016;
    ip.hind_wheel_thick = 0.022;
    
    ip.knee_bent = 3.7895 * 0.63333;
    ip.abad_lb = zeros(4,1); % lower bounds of abad joint angle 
    ip.abad_ub = zeros(4,1); % upper bounds of abad joint angle 
    ip.hhip_lb = [-pi/2.0, -pi/2.0, 0, 0]; % lower bounds of hhip joint angle 
    ip.hhip_ub = [3.0*pi/2.0, 3.0*pi/2.0, pi, pi]; % upper bounds of hhip joint angle 
    ip.knee_lb = -ones(4,1)*ip.knee_bent; % lower bounds of knee joint angle 
    ip.knee_ub = zeros(4,1); % upper bounds of knee joint angle 

    ip.g = 9.8;    
    ip.T = 0.002;
    
    ip.mu = 0.5;
    ip.index = 0;

    ip.nx = nx; % number of states
    ip.nu = nu; % number of inputs(manipulated variables)
	ip.ny = ny; % number of outputs
    ip.noc = noc; % number of contact point
    
    ip.Udweight = [ones(1, ip.nu-ip.noc*3)*0.1, ones(1, ip.noc*3)*0.01];
    ip.Yweight = [100, 1000, 1000, 100, 70]; % ones(1, ip.ny)*100;
    ip.SlackWeight = 10^5;
    ip.pzmin = 5e-3;
    ip.emin = 1e-6;
    
    ip.Yref = zeros(1, ip.ny);
    
    ip.ContactFzLb = 0;
    ip.ContactFzUb = 60;
    ip.XkHistory = [];
    ip.Xk1History = [];
    ip.UkHistory = [];
    ip.ContactF = [];
    ip.ContactP = [];
    
    ip.ceq = [];
    ip.cineq = [];
    init_ip = ip;
end
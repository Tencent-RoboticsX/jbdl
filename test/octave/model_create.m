function tree = model_create()
%model_create - Description
%
% Syntax: tree = model_create()
%
% create rigid body model

global ip;

tree.NB =  7; % Number of rigid bodies, floatbase + joint = 3 + 4
tree.Nc = 2; % Number of contact points

% Parent of each bodies
tree.parent = [0,1,2,3,3,4,5]; 

% Type of each joints, 0-Revolute, 1-Prismatic
tree.jtype = [1,1,0,0,0,0,0]; 
tree.jaxis = ['x', 'z', 'y', 'y', 'y', 'y', 'y'];

% Xtree：Transformation matrices between body lambda(k) to joint k
% on the body, its joint to its children's joint
% SpatialTransform(rotation_matrix, transition_matrix); 
% Note:
% 1. first rotation then transition
EIdentity = eye(3);
r_zeros = zeros(3,1);
r_base_hhip_fore = [ip.length_body*0.5, 0.0, 0.0]';
r_base_hhip_hind = [-ip.length_body*0.5, 0.0, 0.0]';
r_hhip_knee_fore = [0.0, 0.0, -ip.length_hhip]';
r_hhip_knee_hind = [0.0, 0.0, -ip.length_hhip]';
tree.Xtree{1} = SpatialTransform(EIdentity, r_zeros); % j0 -> j1
tree.Xtree{2} = SpatialTransform(EIdentity, r_zeros); % j1 -> j2
tree.Xtree{3} = SpatialTransform(EIdentity, r_zeros); % j2 -> j3
tree.Xtree{4} = SpatialTransform(EIdentity, r_base_hhip_fore); % j3 -> j4
tree.Xtree{5} = SpatialTransform(EIdentity, r_base_hhip_hind); % j3 -> j5
tree.Xtree{6} = SpatialTransform(EIdentity, r_hhip_knee_fore); % j4 -> j6
tree.Xtree{7} = SpatialTransform(EIdentity, r_hhip_knee_hind); % j5 -> j7

% Bounds of joint angles
tree.j_lb = [zeros(1,3),[ip.hhip_lb(1), ip.hhip_lb(3)],[ip.knee_lb(1), ip.knee_lb(3)]]; % Lower bound of each joint angles
tree.j_ub = [zeros(1,3),[ip.hhip_ub(1), ip.hhip_ub(3)],[ip.knee_ub(1), ip.knee_ub(3)]]; % Upper bound of each joint angles

% I: Inertial matrices about joint of each legs
% RigidBodyInertia(mass, CoM of body, inertia of bady);
% Note:  
% 1. CoM of body: reletive to joint frame
% 2. inertia of bady: reletive to body CoM frame
tree.I{1} = RigidBodyInertia(0, zeros(3,1), zeros(3));
tree.I{2} = RigidBodyInertia(0, zeros(3,1), zeros(3));
tree.I{3} = RigidBodyInertia(ip.Massbase, ip.CoMbase, ip.Inertiabase);
tree.I{4} = RigidBodyInertia(ip.MassForeHhip, ip.CoMForeHhip, ip.InertiaForeHhip);
tree.I{5} = RigidBodyInertia(ip.MassHindHhip, ip.CoMHindHhip, ip.InertiaHindHhip);
tree.I{6} = RigidBodyInertia(ip.MassForeKnee, ip.CoMForeKnee, ip.InertiaForeKnee);
tree.I{7} = RigidBodyInertia(ip.MassHindKnee, ip.CoMHindKnee, ip.InertiaHindKnee);

% Uniform acceleration field
tree.a_grav = [0.0; 0.0; 0.0; 0.0; 0.0; -9.81]; % spatial 

% Link plot
tree.idlinkplot = [3, 3, 4, 5, 6, 7]; % body id for link plot
tree.linkplot{1} = r_base_hhip_fore;
tree.linkplot{2} = r_base_hhip_hind;
tree.linkplot{3} = r_hhip_knee_fore;
tree.linkplot{4} = r_hhip_knee_hind;
tree.linkplot{5} = [0.0, 0.0, -ip.length_knee]';
tree.linkplot{6} = [0.0, 0.0, -ip.length_knee]';

tree.idcontact = [6, 7]; % body id of contact point
tree.contactpoint{1} = [0.0, 0.0, -ip.length_knee]';
tree.contactpoint{2} = [0.0, 0.0, -ip.length_knee]';

% CoM Inertial plot
tree.idcomplot = [3, 4, 5, 6, 7];
tree.Mass{1} = ip.Massbase;
tree.Mass{2} = ip.MassForeHhip;
tree.Mass{3} = ip.MassHindHhip;
tree.Mass{4} = ip.MassForeKnee;
tree.Mass{5} = ip.MassHindKnee;

tree.CoM{1} = ip.CoMbase;
tree.CoM{2} = ip.CoMForeHhip;
tree.CoM{3} = ip.CoMHindHhip;
tree.CoM{4} = ip.CoMForeKnee;
tree.CoM{5} = ip.CoMHindKnee;

tree.Inertia{1} = ip.Inertiabase;
tree.Inertia{2} = ip.InertiaForeHhip;
tree.Inertia{3} = ip.InertiaHindHhip;
tree.Inertia{4} = ip.InertiaForeKnee;
tree.Inertia{5} = ip.InertiaHindKnee;

end
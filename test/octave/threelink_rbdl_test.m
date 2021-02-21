addpath(genpath('mRBDL'))
mass_1 = 0.2;
mass_2 = 0.5;
mass_3 = 0.15;

length_1 = 0.06;
length_2 = 0.25;
length_3 = 0.25;

com_1 = [0, 0, 0];
com_2 = [0.035264, 0, 0];
com_3 = [0.129489, 0, 0];

inertia_1 = diag([0.000162754, 0.000162754, 0.00025765]);
inertia_2 = diag([0.000169421, 0.00224942, 0.00234432]);
inertia_3 = diag([0.0000153794, 0.00402557, 0.00402725]);

E_1 = [0, 1, 0;
       0, 0, 1;
       1, 0, 0];
E_2 = [0, -1,  0;
       0,  0, -1;
       1,  0,  0];
E_3 = [1, 0, 0;
       0, 1, 0;
       0, 0, 1];
r_1 = [0, 0, 0.3];
r_2 = [length_1, 0, 0.08];
r_3 = [length_2, 0, 0];
X_T_1 = SpatialTransform(E_1, r_1);
X_T_2 = SpatialTransform(E_2, r_2); 
X_T_3 = SpatialTransform(E_3, r_3); 

model.NB = 3;
model.parent = [0,1,2,3];
model.jtype = [0,0,0];
model.Xtree{1} = X_T_1;
model.Xtree{2} = X_T_2;
model.Xtree{3} = X_T_3;

model.I{1} = RigidBodyInertia( mass_1, com_1, inertia_1 );
model.I{2} = RigidBodyInertia( mass_2, com_2, inertia_2 );
model.I{3} = RigidBodyInertia( mass_3, com_3, inertia_3 );

model.a_grav = [0;0;0;0;0;-9.81];

q = [1, 2, 3]';
qdot = [1, 2, 3]';
tau = [1, 2, 3]';
body_id = 3;
point_pos = [0., 0, 0];
body_pos = zeros(3,3);
body_vel = zeros(3,3);
for i = 1:3
    body_pos(:,i) = CalcBodyToBaseCoordinates(model, q, i, point_pos);
    body_vel(:,i) = CalcPointVelocity(model, q, qdot, i, point_pos);
end
body_pos = body_pos'
body_vel = body_vel'
point_pos = [0.25, 0, 0];
foot_pos = CalcBodyToBaseCoordinates(model, q, 3, point_pos)
foot_vel = CalcPointVelocity(model, q, qdot, 3, point_pos)
foot_jacobian = CalcPointJacobian(model, q, 3, point_pos)
foot_JDotQDot = CalcPointAcceleration(model, q, qdot, zeros(3,1), 3, point_pos)
foot_JDot = CalcPointJacobianDerivative(model, q, qdot, 3, point_pos)
acc_error = foot_JDotQDot - foot_JDot*qdot
H = CompositeRigidBodyAlgorithm(model, q)
C = InverseDynamics(model, q, qdot, zeros(3,1))
qddot_fd = ForwardDynamics(model, q, qdot, tau)
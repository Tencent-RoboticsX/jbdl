function tau = InverseDynamics(model, q, qdot, qddot)
%INVERSEDYNAMICS Summary of this function goes here
%   Detailed explanation goes here

a_grav = model.a_grav;
tau = zeros(model.NB, 1);
for i = 1:model.NB
    [ XJ, S{i} ] = JointModel( model.jtype(i), model.jaxis(i), q(i) );
    vJ = S{i}*qdot(i);
    Xup{i} = XJ * model.Xtree{i};
    if model.parent(i) == 0
        v{i} = vJ;
        avp{i} = Xup{i} * -a_grav;
    else
        v{i} = Xup{i}*v{model.parent(i)} + vJ;
        avp{i} = Xup{i}*avp{model.parent(i)} + S{i}*qddot(i)+ CrossMotionSpace(v{i})*vJ;
    end
    fvp{i} = model.I{i}*avp{i} + CrossForceSpace(v{i})*model.I{i}*v{i};
end

for i = model.NB:-1:1
    tau(i) = S{i}' * fvp{i};
    if model.parent(i) ~= 0
        fvp{model.parent(i)} = fvp{model.parent(i)} + Xup{i}'*fvp{i};
    end
end

end


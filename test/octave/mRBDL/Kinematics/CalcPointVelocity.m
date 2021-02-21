function vel = CalcPointVelocity(model, q, qdot, body_id, point_pos)
%CALCPOINTVELOCITY Summary of this function goes here
%   Detailed explanation goes here

for i = 1:body_id
    [ XJ, S{i} ] = JointModel( model.jtype(i), model.jaxis(i), q(i) );
    vJ = S{i}*qdot(i);
    Xup{i} = XJ * model.Xtree{i};
    if model.parent(i) == 0
        v{i} = vJ;
        X0{i} = Xup{i};
    else
        v{i} = Xup{i}*v{model.parent(i)} + vJ;
        X0{i} = Xup{i}*X0{model.parent(i)};
    end
end
XT_point = Xtrans(point_pos);
X0_point = XT_point * X0{body_id};
vel_spatial = XT_point * v{body_id};
vel = X0_point(1:3,1:3).'*vel_spatial(4:6);
end

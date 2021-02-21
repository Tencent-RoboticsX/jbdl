function J = CalcPointJacobian(model, q, body_id, point_pos)
%CALCPOINTJACOBIAN Summary of this function goes here
%   Detailed explanation goes here
for i = 1:body_id
    [ XJ, S{i} ] = JointModel( model.jtype(i), model.jaxis(i), q(i) );
    Xup{i} = XJ * model.Xtree{i};
    if model.parent(i) == 0
        X0{i} = Xup{i};
    else
        X0{i} = Xup{i}*X0{model.parent(i)};
    end
end
XT_point = Xtrans(point_pos);
X0_point = XT_point * X0{body_id};

j_p = body_id;
BJ = zeros(6, model.NB);
while j_p ~= 0
    Xe{j_p} = X0_point*InverseMotionSpace(X0{j_p});
    BJ(:,j_p) = Xe{j_p}*S{j_p};
    j_p = model.parent(j_p);
end
E0 = X0_point(1:3,1:3)';
J = E0*[zeros(3),eye(3)]*BJ;
end


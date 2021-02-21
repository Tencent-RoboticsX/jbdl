function JDot = CalcPointJacobianDerivative(model, q, qdot, body_id, point_pos)
%CALCPOINTJACOBIANDERIVATIVE Summary of this function goes here
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
v_point = XT_point*v{body_id};

BJ = zeros(6,model.NB);
dBJ = zeros(6,model.NB);
id_p = body_id;
while id_p ~= 0
    if id_p == body_id
        Xe{id_p} = XT_point*Xup{id_p};
        BJ(:,id_p) = XT_point*S{id_p};
        dBJ(:,id_p) = CrossMotionSpace(XT_point*v{id_p} - v_point)*XT_point*S{id_p};
    else
        Xe{id_p} = Xe{id}*Xup{id_p};
        BJ(:,id_p) = Xe{id}*S{id_p};
        dBJ(:,id_p) = CrossMotionSpace(Xe{id}*v{id_p} - v_point)*Xe{id}*S{id_p};
    end
    id = id_p;
    id_p = model.parent(id);
end
X0 = InverseMotionSpace(X0_point);
E0 = [X0(1:3,1:3), zeros(3);
      zeros(3), X0(1:3,1:3)];
dE0 = CrossMotionSpace(X0*v_point)*E0;
E0 = E0(1:3,1:3);
dE0 = dE0(1:3,1:3);
JDot = dE0*[zeros(3),eye(3)]*BJ + E0*[zeros(3),eye(3)]*dBJ;
end


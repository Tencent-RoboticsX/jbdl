function acc = CalcPointAcceleration(model, q, qdot, qddot, body_id, point_pos)
%CALCPOINTACCELERATION Summary of this function goes here
%   Detailed explanation goes here
% 
for i = 1:body_id
    [ XJ, S{i} ] = JointModel( model.jtype(i), model.jaxis(i), q(i) );
    vJ = S{i}*qdot(i);
    Xup{i} = XJ * model.Xtree{i};
    if model.parent(i) == 0
        v{i} = vJ;
        avp{i} = Xup{i} * zeros(6,1);
        X0{i} = Xup{i};
    else
        v{i} = Xup{i}*v{model.parent(i)} + vJ;
        avp{i} = Xup{i}*avp{model.parent(i)} + S{i}*qddot(i)+ CrossMotionSpace(v{i})*vJ;
        X0{i} = Xup{i}*X0{model.parent(i)};
    end
end
E_point = X0{body_id}(1:3,1:3);
XT_point = Xtrans(point_pos);
vel_p = XT_point * v{body_id};
avp_p = XT_point * avp{body_id};
acc = E_point'*avp_p(4:6) + cross(E_point'*vel_p(1:3), E_point'*vel_p(4:6));
% J = CalcPointJacobian(model,q,body_id,point_pos);
% JDot = CalcPointJacobianDerivative(model, q, qdot, body_id, point_pos);
% acc = J*qddot + JDot*qdot;
end


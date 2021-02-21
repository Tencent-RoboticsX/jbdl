function pos = CalcBodyToBaseCoordinates(model, q, body_id, point_pos)
%CALCBODYTOBASECOORDINATES Summary of this function goes here
%   Detailed explanation goes here

for i = 1:body_id
    [ XJ, ~ ] = JointModel( model.jtype(i), model.jaxis(i), q(i) );
    Xup{i} = XJ * model.Xtree{i};
    if model.parent(i) == 0
        X0{i} = Xup{i};
    else
        X0{i} = Xup{i}*X0{model.parent(i)};
    end
end
XT_point = Xtrans(point_pos);
X0_point = XT_point * X0{body_id};
pos = TransformToPosition(X0_point);
end


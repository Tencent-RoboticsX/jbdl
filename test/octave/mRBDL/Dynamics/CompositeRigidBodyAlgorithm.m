function H = CompositeRigidBodyAlgorithm(model, q)
%JOINTSPACEINERTIAMATRIX Summary of this function goes here
%   Detailed explanation goes here
H = zeros(model.NB);

for i = 1:model.NB
    [ XJ, S{i} ] = JointModel( model.jtype(i), model.jaxis(i), q(i) );
    Xup{i} = XJ * model.Xtree{i};
end

IC = model.I;				% composite inertia calculation

for i = model.NB:-1:1
    if model.parent(i) ~= 0
        IC{model.parent(i)} = IC{model.parent(i)} + Xup{i}'*IC{i}*Xup{i};
    end
end

for i = 1:model.NB
    fh = IC{i} * S{i};
    H(i,i) = S{i}' * fh;
    j = i;
    while model.parent(j) > 0
        fh = Xup{j}' * fh;
        j = model.parent(j);
        H(i,j) = S{j}' * fh;
        H(j,i) = H(i,j);
    end
end

end


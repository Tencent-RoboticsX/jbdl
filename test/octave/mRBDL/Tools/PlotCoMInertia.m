function PlotCoMInertia(model, q)
%PlotCoMInertia - Plot CoM point and Inertia on model
%
% Syntax: PlotCoMInertia(model)
%
% Long description
    % Plot CoM 
    hold on;
    pos_com = [];
    num = max(size(model.idcomplot,1), size(model.idcomplot,2));
    for i=1:num
        pos_com(:,i) = CalcBodyToBaseCoordinates(model, q, model.idcomplot(i), model.CoM{i});
        plot3(pos_com(1, i),pos_com(2, i),pos_com(3, i), '*');
    end

    % Plot Inertia
    for i=1:num
        lxyz(:,i) = CalcInertiaCubiod(...
        [model.Inertia{i}(1,1), model.Inertia{i}(2,2), model.Inertia{i}(3,3)]',...
         model.Mass{i});
        PlotInertiaCuboid(lxyz(:,i), pos_com(:,i));
    end
    axis([-0.3 -0.3+0.8 -0.15 -0.15+0.8 -0.1 -0.1+0.8]);
end
function PlotModel(model, q)
%PlotModel - Plot links of model
%
% Syntax: PlotModel(q)
%
% Long description
pos_o = [];
pos_e = [];
num = max(size(model.idlinkplot,1), size(model.idlinkplot,2));
for i=1:num
    pos_o(:,i) = CalcBodyToBaseCoordinates(model, q, model.idlinkplot(i), zeros(3,1));
    pos_e(:,i) = CalcBodyToBaseCoordinates(model, q, model.idlinkplot(i), model.linkplot{i});
end

nc = max(size(model.idcontact,1), size(model.idcontact,2));
for i=1:nc
    pos_contact(:,i) = CalcBodyToBaseCoordinates(model, q, model.idcontact(i), model.contactpoint{i});
end

PlotLink(pos_o, pos_e, num, pos_contact);
    
end
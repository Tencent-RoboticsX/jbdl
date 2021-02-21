function PlotLink(pos_o, pos_e, pos_num, contact_point)
%PlotLink - Description
%
% Syntax: PlotLink(pos_o, pos_e, pos_num, contact_point)
%
% Long description
    hold off;
    for i=1:1:pos_num
        plot3(pos_o(1, i),pos_o(2, i),pos_o(3, i),'o');
        hold on
        plot3([pos_o(1, i),pos_e(1, i)],[pos_o(2, i),pos_e(2, i)],[pos_o(3, i),pos_e(3, i)],'-','linewidth',2);
    end

    [~,h] = size(contact_point);
    for i=1:1:h
        plot3(contact_point(1, i),contact_point(2, i),contact_point(3, i),'.','markersize',5,'color','c');
    end

    grid on;
    box on;
    xlabel x
    ylabel y
    zlabel z  
    % view([0, 0]);  
end
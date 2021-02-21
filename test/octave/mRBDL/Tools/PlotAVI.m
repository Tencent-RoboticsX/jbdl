function PlotAVI(file_name, data, num, model)
    
    ha = figure(1);
    k = 0;
    for j=1:1:num
        q = data(:, j);
        PlotModel(model, q);
        hold on
        title(j);
        % xlim([0,2]);
        % ylim([-2,0]);
        % zlim([0,1]);
        view(0,0);
        axis([-0.9 0.9 -13.0 0.0 -0.9 0.9]);
        
        k=k+1;    
        vf(k)=getframe(ha);
        
        pause(0.2);
    end
    
    v = VideoWriter(file_name);
    open(v);
    writeVideo(v,vf);
    close(v);
    
end
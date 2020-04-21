function cmap = colormap_ripples(interpolate)
% cmap = colormap_ripples(interpolate)
    cmap = [0 109 219;
            109 182 255;
            182 219 255;
            255 255 109;
            219 209 0;
            146 0 0]./255; 
    if interpolate
        [X,Y] = meshgrid(1:3, 1:125);  
        cmap = interp2(X([1,25,50,75,100,125],:), ...
            Y([1,25,50,75,100,125],:),cmap,X,Y);
        cmap=cmap(round(linspace(1,125,64)),:);
    end
end
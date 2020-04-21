function zscores = ripples_zscore_modularities(modularities, null_modularities)
    fq_n = size(modularities{1},1);
    gamma_n = size(modularities{1},2);
    
    zscores = cell(2,1);
    for h=1:2
        zscores{h} = zeros(fq_n,gamma_n);
        for f=1:fq_n
            for g=1:gamma_n
                rval = modularities{h}(f,g);                        % real modularity vals
                nval = squeeze(null_modularities{h}(f,g,:));        % null modularity vals
                zscores{h}(f,g) = (rval - mean(nval))./std(nval);
            end
        end
    end
end
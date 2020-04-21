function rewired_synch_mats = ripples_compute_synch_on_rewired()
    ripples_setup;

    % load PLV data to rewire
    morphed_plv = importdata([FCrp_saveFolder '/group_data/morphed_PLV_allFreqs.mat']);
    rewired_plv_mats = cell(1,2);
    
    % rewire PLV data
    for h=1:2
        rewired_plv_mats{h} = zeros(rwr_n, parc_n, parc_n, fq_n);
        for f=1:fq_n
            plv_mat = morphed_plv{h}(:,:,f); 
            for r=1:rwr_n
                rwr_mat = null_model_und_sign(plv_mat);
                if ~issymmetric(rwr_mat)
                    rwr_mat = triu(rwr_mat,1) + triu(rwr_mat,1)';
                end
                rewired_plv_mats{h}(r,:,:,f) = rwr_mat;
            end     % for r
        end     % for f
    end     % for h

    % compute synch matrices on rewired PLV
    rewired_synch_mats = cell(1,2);
    for h=1:2
        rewired_synch_mats{h} = zeros(rwr_n,parc_n,parc_n,fq_n);
        for f=1:fq_n
            for r=1:rwr_n
                rewired_synch_mats{h}(r,:,:,f) = ripples_compute_synch_matrix(squeeze(rewired_plv_mats{h}(r,:,:,f)));
            end     % for r
        end     % for f
    end     % for h

end
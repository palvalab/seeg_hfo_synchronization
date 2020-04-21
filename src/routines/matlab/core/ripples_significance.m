clear, clc;
ripples_setup;

% load 'true' modules
load([ripples_saveFolder '/modularity_data/modularity_gm' num2str(gammas(1)) 'to' num2str(gammas(end)) '.mat'],...
    'partitions', 'modularities');

% compute synch matrices on rewired PLV data for null modularity computation
rewired_synch_mats = ripples_compute_synch_on_rewired();

% compute null modularity on ensemble of rewired synch matrices
null_modularities = cell(1,2);
for h=1:2
    null_modularities{h} = zeros(fq_n,gamma_n,rwr_n);
    for f=1:fq_n
        for r=1:rwr_n
            rwr_synch_mat = squeeze(rewired_synch_mats{h}(r,:,:,f));
            % compute modules on rewired ensemble
            for g=1:gamma_n
                [~, null_modularities{h}(f,g,r)] = ripples_compute_modularity(rwr_synch_mat, gammas(g));
            end
        end     % for r
    end     % for f
end     % for h

fname = ['null_modularity_' num2str(rwr_n) 'rewirings_gm' num2str(gammas(1)) 'to' num2str(gammas(end)) '.mat'];
save([ripples_saveFolder '/significance_data/' fname], 'null_mod');

% compute z-scored modularity from 'true' data against null distribution
zscored_mod = ripples_zscore_modularities(modularities, null_modularities);
fname = ['zscored_modularity_' num2str(rwr_n) 'rewirings_gm' num2str(gammas(1)) 'to' num2str(gammas(end)) '.mat'];
save([ripples_saveFolder '/significance_data/' fname], 'zscored_mod');

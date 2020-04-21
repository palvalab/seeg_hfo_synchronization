clear; clc;
ripples_setup;

% load bootstrapped data
load([ripples_saveFolder '/group_data/morphed_PLV_' num2str(nBoots) 'bootstraps_allFreqs.mat'],...
    'boot_morphed_plv');

% compute synch matrices on bootstrapped data
boot_synch_mats = cell(2,1);
for h=1:2
    boot_synch_mats{h} = zeros(nBoots, parc_n, parc_n, fq_n);
    for f=1:fq_n
        for b=1:nBoots
            boot_synch_mats{h}(b,:,:,f) = ripples_compute_synch_matrix(squeeze(boot_morphed_plv{h}(:,:,b,f)));
        end     % for b
    end     % for f
end     % for h

% load 'true' modules
load([ripples_saveFolder '/modularity_data/modularity_gm' num2str(gammas(1)) 'to' num2str(gammas(end)) '.mat'],...
    'partitions', 'modularities');

% compute stability of modules against bootstraps
stability = cell(2,1); null_stability = cell(2,1);
for h=1:2
    stability{h} = zeros(parc_n, fq_n, gamma_n, nBoots);
    null_stability{h} = zeros(parc_n, fq_n, gamma_n, nBoots, perm_n);

    for b=1:nBoots
        for f=1:fq_n
            % get true modules
            true_modules = squeeze(partitions{h}(:,f,:));

            % get synch matrix for bootstrapped data
            boot_synch_mat = squeeze(boot_synch_mats{h}(b,:,:,f));
            
            for g=1:gamma_n
                % get 'true' co-assignment matrix
                true_comat = ripples_compute_coassignments(true_modules(:,g));

                % compute bootstraps' modularity and coassignment matrix
                [boot_modules, ~] = ripples_compute_modularity(boot_synch_mat, gammas(g));
                boot_comat = ripples_compute_coassignments(boot_modules);

                % compare bootstrap's co-assignment vectors with the original ones
                stability{h}(:,f,g,b) = ripples_coassignments_similarity(true_comat, boot_comat);

                % permutate 100 times the bootstrap coassignment matrix
                for ii=1:perm_n
                    perm_comat = zeros(size(boot_comat));
                    for p=1:parc_n
                        perm_comat(p,:) = boot_comat(p, randperm(parc_n));
                    end
                    % compute the similarity between 'true' assignments and permuted ones
                    null_stability{h}(:,f,g,b,ii) = ripples_coassignments_similarity(true_comat, perm_comat);
                end     % for ii
            end     % for g
        end     % for f
    end     % for b
end     % for h

% compute stability of 'true' modules wrt bootstrapped ones
avg_true_stability = cell(2,1); avg_null_stability = cell(2,1);
stability_thr = cell(2,1); stability_msk = cell(2,1); 

% get bootstrap and permutation indices for averaging
boot_id = numel(size(stability{1})); perm_id = numel(size(null_stability{1}));

for h=1:2
    % 'true' stability is average stability over bootstraps
    avg_true_stability{h} = mean(stability{h}, boot_id);
    % 'null' stability per bootstrap is average null stability over permutations
    avg_null_stability{h} = mean(null_stability{h}, perm_id);            
    
    % parcel assignment is stable if its avg stability is > 95th percentile of null stability over bootstraps
    stability_thr{h} = prctile(avg_null_stability{h}, 95, boot_id);
    stability_msk{h} = squeeze((avg_true_stability{h} > stability_thr{h}));
end     % for h

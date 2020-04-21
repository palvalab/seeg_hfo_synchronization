clear; clc;
ripples_setup;

fq_n = 2;

% load morphed PLV data
load([ripples_saveFolder '/group_data/morphed_PLV_allFreqs.mat'], 'morphed_plv');

avgOverFreqs = false;
% if needed, average across given frequency range
if avgOverFreqs
    freq_ranges = {[150 210]};
    fq_n = numel(freq_ranges);
   
    allfqs_morphed_plv = morphed_plv;
    morphed_plv = cell(1,2);
    for hh=1:2
        morphed_plv{hh} = zeros(parc_n,parc_n,fq_n);
        for fr = 1:fq_n
            a = find(freq_nums==freq_ranges{fr}(1));
            b = find(freq_nums==freq_ranges{fr}(2));
            morphed_plv{hh}(:,:,fr) = mean(allfqs_morphed_plv{hh}(:,:,a:b),3);
        end
    end
end

% compute synch matrix (correlation of connectivity patterns) for each hemi and freq
synch_mats = cell(1,2);
for h=1:2
    synch_mats{h} = zeros(parc_n,parc_n,fq_n);
    for f=1:fq_n
        synch_mats{h}(:,:,f) = ripples_compute_synch_matrix(squeeze(morphed_plv{h}(:,:,f)));
    end
end
if avgOverFreqs
    save([ripples_saveFolder '/group_data/morphed_PLV_averaged_over_freqs.mat'], 'morphed_plv');
    fname = 'synch_matrices_averaged_over_freqs.mat';
else
    fname = 'synch_matrices_allFreqs.mat';
end
save([ripples_saveFolder '/group_data/' fname], 'synch_mats');

% output data setup
partitions = cell(1,2); modularities = cell(1,2);

for h=1:2
    % output data setup
    partitions{h} = zeros(parc_n,fq_n,gamma_n);
    modularities{h} = zeros(fq_n,gamma_n);

    for f=1:fq_n
        synch_mat = squeeze(synch_mats{h}(:,:,f));
        for g=1:gamma_n
            % compute modules and modularity for given synch matrix
            [partitions{h}(:,f,g), modularities{h}(f,g)] = ripples_compute_modularity(synch_mat, gammas(g));
        end     % for g
    end     % for f
end     % for h

% save final data
if avgOverFreqs
    fname = ['modularity_gm' num2str(gammas(1)) 'to' num2str(gammas(end)) '_averaged_over_freqs.mat'];
else
    fname = ['modularity_gm' num2str(gammas(1)) 'to' num2str(gammas(end)) '.mat'];
end
save([ripples_saveFolder '/modularity_data/' fname], 'partitions', 'modularities');

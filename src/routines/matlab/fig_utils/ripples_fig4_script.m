clear; clc;
ripples_setup;

cmap = colormap_ripples(true);
freq_range = [150 210]; fr = 1; gm = find(gammas==1.25);

%% a. frequency-wise similarity
% load morphed PLV data
load([ripples_saveFolder '/group_data/morphed_PLV_allFreqs.mat'], 'morphed_plv');

% compute freq-wise similarity
freq_similarity = cell(2,1);
for h=1:2
    freq_similarity{h} = zeros(fq_n, fq_n);
    for f1=1:fq_n
        plv1 = squeeze(morphed_plv{h}(:,:,f1));
        for f2=1:fq_n
            plv2 = squeeze(morphed_plv{h}(:,:,f2));
            msk = (plv1~=0 && plv2~=0);
            freq_similarity{h}(f1, f2) = corr(plv1(msk), plv2(msk), 'type', 'Spearman');
        end
    end
end

% show
fig_4a = figure; colormap(cmap);
for h=1:2
    subplot(1,2,h); imagesc(flipud(freq_similarity{h}));
    
    xlabel('Frequency (Hz)'), ylabel('Frequency (Hz)');
    xticks(1:fq_n), xticklabels(freq_vect);
    yticks(1:fq_n), yticklabels(flipud(freq_vect));
    
    c = colorbar; ylabel(c, 'PLV correlation');
end

%% b. z-scored modularity across frequencies
% load zscored modularity
fname = ['zscored_modularity_' num2str(rwr_n) 'rewirings_gm' num2str(gammas(1)) 'to' num2str(gammas(end)) '.mat'];
load([ripples_saveFolder '/significance_data/' fname], 'zscored_mod');

fig_4b = figure; hold on;
for h=1:2
    avg_zscore = mean(zscored_mod{h}, 2);
    plot(avg_zscore, 'LineWidth', 1.7);
end
hold off; legend(hemis); xticks(1:fq_n); xticklabels(freq_vect);

%% d. assignment panels for given freq range
fname = ['modularity_gm' num2str(gammas(1)) 'to' num2str(gammas(end)) '_averaged_over_freqs.mat'];
load([ripples_saveFolder '/modularity_data/' fname], 'partitions', 'modularities');

fig_4d = figure; colormap(cmap);
for h=1:2
    subplot(1,2,h); imagesc(partitions{h}(:,fr,:));
    
    xticks(1:gamma_n); xticklabels(gammas);
    yticks([1, parc_n]); yticklabels({num2str(parc_n), '1'});
end

%% e. synch matrices for given freq range and gamma, sorted wrt modules
fname = 'synch_matrices_averaged_over_freqs.mat';
load([ripples_saveFolder '/group_data/' fname], 'synch_mats');

fig_4e = figure; colormap(cmap);
for h=1:2
    synch_mat = synch_mats{h}(:,:,fr);
    M = partitions{h}(:,fr,gm);
    [srt_M, srt_ii] = sort(M);
    
    subplot(1,2,h); imagesc(synch_mat(srt_ii, srt_ii));
    
    xticks([]); yticks([]);
    colormap(cmap); c = colorbar; ylabel(c, 'Synchrony Profile Similarity');
end

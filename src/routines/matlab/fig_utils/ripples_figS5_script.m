clear; clc; close all;
ripples_setup;
cmap = colormap_ripples(true);

%% b. nodal strength across freqs
load([ripples_saveFolder '/group_data/morphed_PLV_allFreqs.mat'], 'morphed_plv');

avg_nodal_strengths = cell(2,1);
for h=1:2
    avg_nodal_strengths{h} = squeeze(mean(morphed_plv{h}));
end

fig_S5b = figure; colormap(cmap);
for h=1:2
    subplot(1,2,h); imagesc(flipud(avg_nodal_strengths{h}));
    
    xlabel('Parcels'), ylabel('Frequency (Hz)');
    xticks([1,parc_n]), xticklabels({'1', num2str(parc_n)});
    yticks(1:fq_n), yticklabels(flipud(freq_vect));
    
    c = colorbar; ylabel(c, 'Nodal strength');
end

%% c. nodal strength correlation across freqs
nodal_corr = cell(2,1);
for h=1:2
    nodal_corr{h} = zeros(fq_n, fq_n);
    for f1 = 1:fq_n
        strength1 = squeeze(avg_nodal_strengths{h}(f1,:));
        for f2 = 1:fq_n
            strength2 = squeeze(avg_nodal_strengths{h}(f2,:));
            nodal_corr{h}(f1,f2) = corr(f1', f2', 'type', 'Spearman');
        end
    end
end

fig_S5c = figure; colormap(cmap);
for h=1:2
    subplot(1,2,h); imagesc(flipud(nodal_corr{h}));
    
    xlabel('Frequency (Hz)'), ylabel('Frequency (Hz)');
    xticks(1:fq_n), xticklabels(freq_vect);
    yticks(1:fq_n), yticklabels(flipud(freq_vect));
    
    c = colorbar; ylabel(c, 'Nodal strength correlation');
end

%% d. z-scored modularity across freqs and gamma
fname = ['zscored_modularity_' num2str(rwr_n) 'rewirings_gm' num2str(gammas(1)) 'to' num2str(gammas(end)) '.mat'];
load([ripples_saveFolder '/significance_data/' fname], 'zscored_mod');

% fig setup
fig_S5d = figure; colormap(cmap);
for h=1:2
    subplot(1,2,h); imagesc(flipud(zscored_mod{h}));
    
    xlabel('Resolution parameter'), ylabel('Frequency (Hz)');
    xticks(1:gamma_n), xticklabels(gammas);
    yticks(1:fq_n), yticklabels(flipud(freq_vect));
    
    c = colorbar; ylabel(c, 'Modularity (z-scored)');
end

%% e. PLV matrices for given freq range
load([ripples_saveFolder '/group_data/morphed_PLV_averaged_over_freqs.mat'], 'morphed_plv_avg');
fig_S5e = figure;
for h=1:2
    plv_mat = squeeze(morphed_plv_avg{h}(:,:,fr));
    plv_mat(plv_mat==0) = NaN;
    subplot(1,2,h); imagesc(plv_mat, 'AlphaData', ~isnan(plv_mat));
    
    xlabel('Parcels'), ylabel('Parcels');
    xticks([1,parc_n]), xticklabels({'1', num2str(parc_n)});
    yticks([1,parc_n]), yticklabels({num2str(parc_n), '1'});
    
    c = colorbar; ylabel(c, 'PLV');
end

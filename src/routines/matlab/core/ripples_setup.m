% dirs information setting
thisFilePath = mfilename('fullpath');
ripples_codeFolder = fileparts(thisFilePath);
ripples_baseFolder = [ripples_codeFolder '/../..'];
ripples_dataFolder = [ripples_baseFolder '/data/ripples_data'];
ripples_saveFolder = [ripples_baseFolder '/data/ripples_data/saved_data'];

% load subjects list and remove 'non-valid' subjects (#18, #36 and #57)
all_sbjs = readtable([ripples_dataFolder '/ripples_aux_data/Ripples_subject_agg_cluster_labels.csv']);
sbjs_to_remove = [18; 36; 57];  % known non-valid subjects

% assignation of each subject to its cluster
sbj_clust_ids = cell(max(all_sbjs.subject_cluster), 1);
for c = 1:max(all_sbjs.subject_cluster)
    sbj_clust_ids{c} = find(all_sbjs.subject_cluster == c);
end

% definition of frequencies
freq_nums = importdata([ripples_dataFolder '/ripples_aux_data/Ripples_freqs.txt']);
freq_vect = cellfun(@num2str, num2cell(freq_nums), 'UniformOutput', false);
fq_n = numel(freq_vect);

% definition of gamma range for modularity algorithm
gammas = 1:.05:1.5;
gamma_n = numel(gammas);

rwr_n = 100;    % # of rewirings for significance analysis
nBoots = 100;   % # of bootstraps for stability analysis
perm_n = 100;   % # of permutations for stability analysis

% cosmetics
hemis = {'LH', 'RH'}; parc_n = 50;
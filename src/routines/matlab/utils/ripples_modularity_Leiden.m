function [M, Q1] = ripples_modularity_Leiden(mat, gm, iters)
% usage: [M, Q1] = ripples_modularity_Leiden(mat, gm, iters)
% NB: mat MUST BE SYMMETRICAL

    leiden_dirpath = [getenv('HOME') '/Programs/NetworkClusteringLeiden'];
    jarpath = [leiden_dirpath '/RunNetworkClustering.jar'];
    
    % get upper triangular matrix (avoids duplicate edges)
    mat = triu(mat);
    
    % get edge list (0-indexed) and corresponding weights
    edge_idxs = find(mat>0);
    edge_wgts = mat(edge_idxs);
    [r, c] = ind2sub(size(mat), edge_idxs);
    edge_list = [r-1 c-1 edge_wgts];

    % create unique filenames
    tmp_fname = tempname(leiden_dirpath);
    edge_fname = [tmp_fname '_edgesW.txt'];
    
    % write edge list to file
    dlmwrite(edge_fname, edge_list, '\t');
    
    % run jar on edge list
    [status, cmdout] = system(['java -jar ' jarpath ' -q modularity -r ' num2str(gm)...
        ' -i ' num2str(iters) ' -w ' edge_fname]);
    if status~=0
        error(['[ERROR ripples_modularity_Leiden] - ' cmdout]);
    else
        % split output in lines
        split_out = strsplit(cmdout, '\n');
        % find line matching modularity line and extract value
        res = cellfun(@(str) sscanf(str, 'Quality function equals %f.'), split_out, 'uni', false);
        % modularity = value in only cell where result is not empty
        Q1 = res{~cellfun(@isempty, res)};
        
        % find last line before partition
        l_id = find(strcmp(split_out, 'Writing final clustering to standard output.'));
        % get partition lines (last line is empty b/c of \n), convert strings to numbers and cell to array
        clust_dt = cell2mat(cellfun(@(el) str2num(el), split_out(l_id+1:end-1)', 'UniformOutput', false));
        % first column is node ids, second is module idces
        nodes = clust_dt(:,1);
        M = clust_dt(:,2)+1;

        % check for isolated nodes: if nodes are missing from M, add them w/ module idx 0
        nodes_n = size(mat, 1);
        if ~isempty(setdiff(0:nodes_n-1, nodes))
            missing_nodes = setdiff(0:nodes_n-1, nodes)'; % column vector
            clust_dt = [nodes           M;...
                        missing_nodes   zeros(size(missing_nodes))];
            % sort on nodes idxs to reorder (nodes in original order, zeros are put beside isolated nodes)
            [~, idxs] = sort(clust_dt(:,1));
            clust_dt = clust_dt(idxs, :);
            M = clust_dt(:,2);
        end
        
        % delete unique file created
        delete(edge_fname);
    end
end
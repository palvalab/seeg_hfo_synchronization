function coassignment_mat = ripples_compute_coassignments(modules)
    mods_n = max(modules);
    parc_n = length(modules);
    coassignment_mat = zeros(parc_n, parc_n, mods_n);   % one matrix per module
    
    for m = 1:mods_n
        parcs_in_M = double(modules == m);
        if size(parcs_in_M,1)==1
            coassignment_mat(:,:,m)= parcs_in_M' * parcs_in_M;
        else
            coassignment_mat(:,:,m)= parcs_in_M * parcs_in_M';
        end
    end
    coassignment_mat = sum(coassignment_mat, 3);        % sum over modules
end
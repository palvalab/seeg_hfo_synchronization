function [M, Q1] = ripples_compute_modularity(mat, gm)
    parc_n = size(mat,1);
    
    M  = 1:parc_n;
    Q0 = -1; Q1 = 0;
    while Q1-Q0>1e-5
        Q0 = Q1;
        [M, Q1] = w_modularity_Leiden(mat, gm, 100);
    end
end
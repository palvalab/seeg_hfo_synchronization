function synch_mat = ripples_compute_synch_matrix(morphed_plv_data)
    parc_n = size(morphed_plv_data, 1);
    
    corr_mat = zeros(parc_n,parc_n);
    for p1=1:parc_n
        % get parcel's row in morphed PLV matrix
        p1_vec = morphed_plv_data(:,p1);
        for p2=(p1+1):parc_n
            % get parcel's row in morphed PLV matrix
            p2_vec = morphed_plv_data(:,p2);
            
            % get only edges ~= 0 in both rows
            msk = (p1_vec~=0) & (p2_vec~=0);
            
            if sum(msk)>1
                % compute rows correlation
                corr_mat(p1,p2) = corr(p1_vec(msk), p2_vec(msk), 'Type', 'Spearman');
            else
                error('Insufficient data');
            end
        end     % for p2
    end     % for p1
    % symmetrize
    synch_mat = triu(corr_mat,1) + triu(corr_mat,1)';
end
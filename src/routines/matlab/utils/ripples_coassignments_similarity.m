function similarity = ripples_coassignments_similarity(v1, v2)
% sml = ripples_coassignments_similarity(v1, v2)
% v1 and v2 may be either coassignment vectors or matrices
    np = size(v1,1);
    if size(v1,2) ~= 1                      % matrices => sum over rows for parcels
        sum1s = sum(and(v1, v2), 2) - 1;    % removing contribution from parcel p
        sum0s = sum(and(~v1, ~v2), 2);
    else
        sum1s = sum(and(v1, v2)) - 1;
        sum0s = sum(and(~v1, ~v2));
    end
    similarity = (sum1s + sum0s)/(np-1);
end
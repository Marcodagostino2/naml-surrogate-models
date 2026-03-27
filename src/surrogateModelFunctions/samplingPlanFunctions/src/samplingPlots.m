function samplingPlots(x)
    %% Pairwise distance distribution
    D = squareform(pdist(x));  % full pairwise distances
    D = D(triu(true(size(D)),1));  % upper triangle

    figure('Name','Pairwise Distance Distribution','Color','w');
    histogram(D, 40,'Normalization','pdf');
    xlabel('Distance');
    ylabel('p(d)');
    title('Distribution of Pairwise Distances');
    grid on;
end

function [augR, augL, flag] = buildCovarianceMatrixGEK(x, theta, p, lambda, dyDirs)
% buildCovarianceMatrixGEK  Construct the GEK block covariance matrix.
    n = size(x, 1);
    m = numel(dyDirs);
    augN = n * (1 + m);
    
    % Base R
    D = zeros(n, n);
    for k = 1:size(x, 2)
        D = D + theta(k) * (abs(x(:, k) - x(:, k)') .^ p(k));
    end
    R = exp(-D);
    augR = zeros(augN, augN);
    augR(1:n, 1:n) = R + eye(n) * lambda;
    
    % Precompute dx and abs(dx)
    dx = cell(1, size(x, 2));
    absDx = cell(1, size(x, 2));
    for k = 1:size(x, 2)
        dx{k} = x(:, k) - x(:, k)';
        absDx{k} = abs(dx{k});
    end
    
    % 1st Derivatives: Cov(y, dy)
    dRDx = cell(1, m);
    for i = 1:m
        k = dyDirs(i);
        term = -theta(k) * p(k) * (absDx{k}.^(p(k)-1)) .* sign(dx{k});
        dRDx{i} = R .* term;
        
        idx = n + (i-1)*n + 1 : n + i*n;
        augR(1:n, idx) = -dRDx{i};
        augR(idx, 1:n) = dRDx{i};
    end
    
    % 2nd Derivatives: Cov(dy, dy)
    for i = 1:m
        k = dyDirs(i);
        idxRow = n + (i-1)*n + 1 : n + i*n;
        for j = i:m
            l = dyDirs(j);
            idxCol = n + (j-1)*n + 1 : n + j*n;
            
            if k == l
                abs_k = absDx{k};
                mask0 = (abs_k == 0);
                abs_k(mask0) = 1; % Temp fix for 0^(p-2)
                
                term2 = theta(k) * p(k) * (p(k)-1) * (abs_k .^ (p(k)-2));
                term2(mask0) = 2 * theta(k); % Exact analytical limit for p=2
                
                Rij = -(dRDx{i}.^2 ./ (R + 1e-12)) + R .* term2;
            else
                Rij = -(dRDx{i} .* dRDx{j}) ./ (R + 1e-12);
            end
            
            augR(idxRow, idxCol) = Rij;
            if i ~= j
                augR(idxCol, idxRow) = Rij';
            end
        end
        augR(idxRow, idxRow) = augR(idxRow, idxRow) + eye(n) * lambda; 
    end
    
    [augL, flag] = chol(augR, 'lower');
end
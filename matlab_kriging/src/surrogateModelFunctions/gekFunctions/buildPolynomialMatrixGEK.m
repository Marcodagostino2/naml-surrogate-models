function fAug = buildPolynomialMatrixGEK(x, polyDegree, dyDirs, fBase)
% buildPolynomialMatrixGEK Extract derivatives of polynomial trends via finite differences
    n = size(x, 1);
    m = numel(dyDirs);
    nPoly = size(fBase, 2);
    
    fAug = zeros(n * (1 + m), nPoly);
    fAug(1:n, :) = fBase;
    
    epsFD = 1e-6; % Central finite differences for exact poly derivatives
    for i = 1:m
        k = dyDirs(i);
        idx = n + (i-1)*n + 1 : n + i*n;
        
        xPlus = x;  xPlus(:, k)  = xPlus(:, k) + epsFD;
        xMinus = x; xMinus(:, k) = xMinus(:, k) - epsFD;
        
        fPlus  = buildPolynomialMatrix(xPlus, polyDegree);
        fMinus = buildPolynomialMatrix(xMinus, polyDegree);
        
        fAug(idx, :) = (fPlus - fMinus) / (2 * epsFD);
    end
end
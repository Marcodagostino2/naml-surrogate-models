function psiCross = crossCorrMatrix(x1, x2, theta, p)
    % crossCorrMatrix  Compute the full cross-correlation matrix.
    
    d = size(x1, 2);
    
    % Reshape arrays into 3D tensors:
    % x1    -> [n1, 1,  d]
    % x2    -> [1,  n2, d]
    % theta -> [1,  1,  d]
    % p     -> [1,  1,  d]
    x13D = reshape(x1, [], 1, d);
    x23D = reshape(x2, 1, [], d);
    th3D = reshape(theta, 1, 1, d);
    p3D  = reshape(p, 1, 1, d);
    
    % Compute the full 3D distance tensor and sum along the 3rd dimension (d).
    % The 'abs(x1_3D - x2_3D)' automatically expands to [n1, n2, d].
    exponent = sum(th3D .* (abs(x13D - x23D) .^ p3D), 3);
    
    % Apply exponential
    psiCross = exp(-exponent);
end


function psiCross = crossCorrMatrix(x1, x2, theta, p)
    % crossCorrMatrix  Compute the full cross-correlation matrix.
    
    d = size(x1, 2);
    
    % Reshape arrays into 3D tensors:
    % x1    -> [n1, 1,  d]
    % x2    -> [1,  n2, d]
    % theta -> [1,  1,  d]
    % p     -> [1,  1,  d]
    x1_3D = reshape(x1, [], 1, d);
    x2_3D = reshape(x2, 1, [], d);
    th_3D = reshape(theta, 1, 1, d);
    p_3D  = reshape(p, 1, 1, d);
    
    % Compute the full 3D distance tensor and sum along the 3rd dimension (d).
    % The 'abs(x1_3D - x2_3D)' automatically expands to [n1, n2, d].
    exponent = sum(th_3D .* (abs(x1_3D - x2_3D) .^ p_3D), 3);
    
    % Apply exponential
    psiCross = exp(-exponent);
end


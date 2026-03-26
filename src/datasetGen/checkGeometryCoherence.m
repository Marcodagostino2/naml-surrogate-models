function coherent = checkGeometryCoherence(geometry1, geometry2)
    % checkGeometryCoherence: Validates that two geometries match.

    % Compare against HF values extracted from CSV
    fieldsToCompare = fieldnames(geometry1.values);
    tol = 1e-4; % Tolerance for floating-point comparison
    mismatches = {};
    coherent = true; % Assume coherence until a mismatch is found

    for i = 1:length(fieldsToCompare)
        f = fieldsToCompare{i};
        valLF = geometry1.values.(f);
        
        if isfield(geometry2.values, f)
            valHF = geometry2.values.(f);
            
            if isnan(valHF)
                warning('GeometryCoherence:NaNValue', 'Field "%s" is NaN in the 2nd geometry. Skipping comparison.', f);
            elseif abs(valLF - valHF) > tol
                % Format the mismatch string for the error report
                mismatches{end+1} = sprintf('  - %s: LF = %.4f | HF = %.4f', f, valLF, valHF); %#ok<AGROW> 
            end
        else
            warning('GeometryCoherence:MissingField', 'Field "%s" is missing from the 2nd geometry.', f);
        end
    end
    
    % Report results
    if ~isempty(mismatches)
        warningMsg = strjoin(mismatches, '\n');
        warning('GeometryCoherence:Mismatch', ...
              'Geometry coherence check FAILED! Differences found between geometry):\n%s\nAborting to prevent training on mixed datasets.', warningMsg);
        coherent = false;
    else
        fprintf('  [OK] Geometry coherence check passed.\n');
    end
end

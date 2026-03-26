function geometryCFD = extractGeometryMetadata(cfdValid)
% extractGeometryMetadata  Reads geometric parameters from the first valid CSV row.

    geometryCFD = struct('values', struct(), 'units', struct());
    
    if isempty(cfdValid)
        error("extractCFDDataset:NoValidData", "No valid CFD data to extract geometry.");
    end

    % Define expected parameters and their units
    geomFields = {
        'sim_xcgImposed', '[m]';
        'geom_rocket_diameter', '[mm]';    'geom_rocket_Lnose', '[mm]';
        'geom_rocket_noseP', '[-]';        'geom_rocket_noseC', '[-]';
        'geom_rocket_Ltot', '[mm]';        'geom_rocket_Lboat', '[mm]';
        'geom_rocket_Dboat', '[mm]';       'geom_rocket_bezierX1', '[mm]';
        'geom_rocket_bezierX2', '[mm]';    'geom_rocket_bezierY2', '[mm]';
        'geom_fins_RootChord', '[mm]';     'geom_fins_FreeChord', '[mm]';
        'geom_fins_Height', '[mm]';        'geom_fins_DeltaXFreeChord', '[mm]';
        'geom_fins_AxialRelativeDistance', '[mm]'; 'geom_fins_SemiThickness', '[mm]';
        'geom_abk_theoricAngle', '[deg]';  'geom_abk_effectiveAngle', '[deg]';
        'geom_abk_maxExtraction', '[mm]';  'geom_abk_width', '[mm]';
        'geom_abk_relativePosition', '[mm]'
    };

    % Use the first valid row for constant geometric parameters
    firstRow = cfdValid(1, :);
    csvVars  = cfdValid.Properties.VariableNames;

    for i = 1:size(geomFields, 1)
        fName = geomFields{i, 1};
        fUnit = geomFields{i, 2};
        
        if ismember(fName, csvVars)
            geometryCFD.values.(fName) = firstRow.(fName);
        else
            warning('extractCFDDataset:MissingGeometry', ...
                    'Geometry field "%s" missing. Assigning NaN.', fName);
            geometryCFD.values.(fName) = NaN;
        end
        geometryCFD.units.(fName) = fUnit;
    end
end

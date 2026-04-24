function exportFigure(fig, baseName, expPath, expFmt)
arguments
    fig
    baseName (1,1) string
    expPath  (1,1) string = ""
    expFmt   (1,1) string = "png"
end
if strlength(expPath) == 0, return; end
if ~exist(expPath, "dir"), mkdir(expPath); end
fileName = fullfile(expPath, sprintf("%s.%s", baseName, lower(expFmt)));
if lower(expFmt) == "pdf"
    exportgraphics(fig, fileName, "ContentType", "vector");
else
    exportgraphics(fig, fileName, "Resolution", 300);
end
end

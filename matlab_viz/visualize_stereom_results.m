% =========================================================================
%  visualize_stereom_results.m
%
%  Visualizes the output of the Python `stereom` pipeline
%  (stereom_python/) from a `results.mat` file written by
%  `stereom.matlab_export.export_matlab` -- produced automatically when
%  `stereom run` is called with `--export-mat`, or afterwards via
%  `stereom export-mat --results-dir <dir>`.
%
%  This is deliberately a single, cell-mode (%%) script in the same style
%  as the original toolbox's main_*.m analysis scripts: set the file path
%  in Section 0, then run section by section (Ctrl+Enter in MATLAB/the
%  MATLAB extension) or the whole thing at once. Every plotting call is a
%  LOCAL FUNCTION at the bottom of this file -- nothing here recomputes
%  any analysis result; it only reads what stereom already computed.
%
%  Coordinate convention: nodes.i / nodes.j / nodes.k are the three ARRAY
%  dimensions of the analyzed volume, i.e. exactly `config.axes` from the
%  Python side (default ('L','C','R') -- longitudinal, circumferential,
%  radial). They are plotted here as plain x/y/z; if you set a different
%  `axes` mapping in Python, relabel the axis titles below to match.
% =========================================================================

%% Section 0: point this at your results.mat and load it
clearvars
close all
clc

% EDIT THIS: full path to the results.mat written by `stereom run ... --export-mat`
resultsFile = 'path\to\your_results\results.mat';
data = load(resultsFile);

% If the ORIGINAL MATLAB toolbox is on your path (or its `toolbox\plot`
% folder is added below), FigureFormat.m is used for the camera/axis
% styling; otherwise a small built-in equivalent (localFigureFormat, at
% the bottom of this file) is used instead. Uncomment and edit if you
% want the original toolbox's exact styling:
% addpath(genpath('path\to\Quantitative-3D-structural-analysis-of-the-cellular-microstructures-master\toolbox'));

useOriginalFigureFormat = (exist('FigureFormat', 'file') == 2);

fprintf('Loaded %s\n', resultsFile);
fprintf('  %d nodes, %d branches\n', numel(data.nodes.id), numel(data.branches.id));
if isfield(data, 'rings')
    fprintf('  %d rings\n', numel(data.rings.id));
end
if isfield(data, 'chains')
    fprintf('  %d chains\n', numel(data.chains.id));
end

%% Section 1: 3D network, branches colored by mean thickness (radius)
figure('Name', 'Network - colored by thickness');
plotNetworkColoredByValue(data, data.branches.t_mean_radius_um, 'Mean radius (\mum)', useOriginalFigureFormat);

%% Section 2: 3D network, branches colored by length
figure('Name', 'Network - colored by length');
plotNetworkColoredByValue(data, data.branches.length_um, 'Branch length (\mum)', useOriginalFigureFormat);

%% Section 3: 3D network, branches colored by tortuosity
figure('Name', 'Network - colored by tortuosity');
plotNetworkColoredByValue(data, data.branches.tortuosity, 'Tortuosity', useOriginalFigureFormat);

%% Section 4: node-type (degree) distribution
figure('Name', 'Node type distribution');
degrees = data.nodes.degree;
edges = (min(degrees) - 0.5):1:(max(degrees) + 0.5);
histogram(degrees, edges, 'Normalization', 'probability', 'FaceColor', [0.20 0.45 0.80]);
xlabel('Node degree (N-type)'); ylabel('Frequency');
title(sprintf('Node type distribution (n=%d)', numel(degrees)));
set(gcf, 'color', 'white'); box on

fprintf('\nNode degree breakdown:\n');
for d = unique(degrees)'
    fprintf('  N-%d: %d (%.1f%%)\n', d, sum(degrees == d), 100 * mean(degrees == d));
end

%% Section 5: branch length, distance and thickness histograms
figure('Name', 'Branch length / distance');
histogram(data.branches.length_um, 30, 'Normalization', 'probability', ...
    'DisplayStyle', 'stairs', 'LineWidth', 1.5, 'EdgeColor', [0.85 0.33 0.10]);
hold on
histogram(data.branches.distance_um, 30, 'Normalization', 'probability', ...
    'DisplayStyle', 'stairs', 'LineWidth', 1.5, 'EdgeColor', [0.20 0.45 0.80]);
legend('curved length l_{ij}', 'node-to-node distance l_{o,ij}');
xlabel('Length (\mum)'); ylabel('Frequency'); box on
set(gcf, 'color', 'white')

figure('Name', 'Branch thickness (radius)');
validThickness = data.branches.t_mean_radius_um(~isnan(data.branches.t_mean_radius_um));
histogram(validThickness, 30, 'Normalization', 'probability', 'FaceColor', [0.30 0.70 0.30]);
xlabel('Mean branch radius (\mum)'); ylabel('Frequency');
title(sprintf('Mean radius: %.2f \\pm %.2f \\mum (n=%d)', ...
    mean(validThickness), std(validThickness), numel(validThickness)));
set(gcf, 'color', 'white'); box on

%% Section 6: branch orientation polar plot (theta = radius, omega = angle)
% Paper convention (NOT the original MATLAB code's swapped naming, spec
% Sec 2.4): theta = misorientation from the L axis (0-90 deg, radial
% coordinate here), omega = in-plane azimuth from the R axis (0-360 deg,
% angular coordinate here).
figure('Name', 'Branch orientation');
polarplot(deg2rad(data.branches.omega_deg), data.branches.theta_deg, 'o', ...
    'MarkerSize', 4, 'MarkerFaceColor', [0.20 0.45 0.80], 'MarkerEdgeColor', 'none');
rticks(0:30:90); rlim([0 90]);
title('Branch orientation (\theta = radius, \omega = angle)');
set(gcf, 'color', 'white')

%% Section 7: inter-branch angle distribution (N-3 and N-4 nodes)
if isfield(data, 'angles')
    nodeDegreeByRow = zeros(numel(data.nodes.row), 1);
    nodeDegreeByRow(data.nodes.row) = data.nodes.degree;
    angleNodeDegree = nodeDegreeByRow(data.angles.node_row);

    figure('Name', 'Inter-branch angles');

    subplot(1, 2, 1)
    a3 = data.angles.angle_deg(angleNodeDegree == 3);
    if ~isempty(a3)
        histogram(a3, 20:10:180, 'Normalization', 'probability', 'FaceColor', [0.20 0.45 0.80]);
    end
    xlabel('Inter-branch angle (deg)'); ylabel('Frequency');
    title(sprintf('N-3 nodes (mean %.1f\\circ)', mean(a3)));
    box on

    subplot(1, 2, 2)
    a4 = data.angles.angle_deg(angleNodeDegree == 4);
    if ~isempty(a4)
        histogram(a4, 20:10:180, 'Normalization', 'probability', 'FaceColor', [0.85 0.33 0.10]);
    end
    xlabel('Inter-branch angle (deg)'); ylabel('Frequency');
    title(sprintf('N-4 nodes (mean %.1f\\circ)', mean(a4)));
    box on
    set(gcf, 'color', 'white')
else
    fprintf('No angles struct in results.mat (ring/chain-only export?) -- skipping Section 7.\n');
end

%% Section 8: rings -- size histogram and 3D polygons colored by ring size
if isfield(data, 'rings') && numel(data.rings.id) > 0
    figure('Name', 'Ring size distribution');
    ringSizes = data.rings.size;
    edges = (min(ringSizes) - 0.5):1:(max(ringSizes) + 0.5);
    histogram(ringSizes, edges, 'FaceColor', [0.60 0.40 0.80]);
    xlabel('Ring size (number of branches)'); ylabel('Count');
    title(sprintf('%d rings detected', numel(ringSizes)));
    set(gcf, 'color', 'white'); box on

    figure('Name', 'Ring geometry (colored by ring size)');
    plotRings(data, useOriginalFigureFormat);
else
    fprintf('No rings in results.mat -- skipping Section 8.\n');
end

%% Section 9: chains -- each chain drawn in a distinct color
if isfield(data, 'chains') && numel(data.chains.id) > 0
    figure('Name', 'Chains (aligned branch sequences)');
    plotChains(data, useOriginalFigureFormat);
else
    fprintf('No chains in results.mat -- skipping Section 9.\n');
end

%% Section 10: profile fits -- pooled normalized branch profile (z/L vs t/t_min)
if isfield(data, 'profile_fits') && numel(data.profile_fits.branch_id) > 0
    figure('Name', 'Profile fit summary');
    accepted = logical(data.profile_fits.accepted);
    fprintf('\nProfile fits: %d/%d branches accepted for fitting\n', ...
        sum(accepted), numel(accepted));

    subplot(1, 2, 1)
    histogram(data.profile_fits.b(accepted), 30, 'FaceColor', [0.20 0.45 0.80]);
    xlabel('Quadratic curvature coefficient b (\mum^{-1})'); ylabel('Count');
    title('t(z) = a_0 + b z^2 fit');
    box on

    subplot(1, 2, 2)
    qa = data.profile_fits.quartic_a(accepted);
    qa = qa(~isnan(qa));
    if ~isempty(qa)
        histogram(qa, 30, 'FaceColor', [0.85 0.33 0.10]);
    end
    xlabel('Quartic parameter a'); ylabel('Count');
    title('Paper II: t/t_{min} = (a(z/L)^2+1)^2 fit');
    box on
    set(gcf, 'color', 'white')
else
    fprintf('No profile_fits in results.mat -- skipping Section 10.\n');
end

%% Section 11: overlay the source STL region, transparent, to verify the
%% network sits correctly inside the actual microstructure
% Only present if results.mat came from an STL region extraction
% (`stereom extract-regions` + `stereom run --export-mat`, which copies
% the companion {name}.stl next to results.mat and embeds a
% `stl_overlay` struct telling us how to align it).
if isfield(data, 'stl_overlay')
    figure('Name', 'Network overlaid on source STL (verification)');
    plotNetworkWithSTLOverlay(data, resultsFile, useOriginalFigureFormat);
else
    fprintf('No stl_overlay in results.mat (this volume wasn''t extracted from an STL via `stereom extract-regions`) -- skipping Section 11.\n');
end


% =========================================================================
%  Local functions
% =========================================================================

function pts = branchCenterline(data, branchIdx)
% Full ordered centerline of one branch: node_a -> interior path -> node_b,
% as an Nx3 [i j k] matrix.
    a = data.branches.node_a_row(branchIdx);
    b = data.branches.node_b_row(branchIdx);
    nodeA = [data.nodes.i(a), data.nodes.j(a), data.nodes.k(a)];
    nodeB = [data.nodes.i(b), data.nodes.j(b), data.nodes.k(b)];
    interior = data.branches.path{branchIdx};
    if isempty(interior)
        pts = [nodeA; nodeB];
    else
        pts = [nodeA; interior; nodeB];
    end
end

function plotNetworkColoredByValue(data, colorValue, colorLabel, useOriginalFigureFormat)
% Draws every branch as a line, colored by `colorValue` (one entry per
% branch) via a jet colormap, plus nodes as black dots.
    cmap = jet(256);
    validColor = colorValue;
    validColor(isnan(validColor)) = 0;
    lo = min(validColor); hi = max(validColor);
    if hi <= lo
        hi = lo + 1;
    end

    hold on
    for b = 1:numel(data.branches.id)
        pts = branchCenterline(data, b);
        if isnan(colorValue(b))
            colorIdx = 1;
        else
            colorIdx = 1 + round(255 * (colorValue(b) - lo) / (hi - lo));
            colorIdx = min(max(colorIdx, 1), 256);
        end
        plot3(pts(:,1), pts(:,2), pts(:,3), '-', 'Color', cmap(colorIdx,:), 'LineWidth', 1.2);
    end
    scatter3(data.nodes.i, data.nodes.j, data.nodes.k, 12, 'k', 'filled');
    hold off

    colormap(jet); cb = colorbar; cb.Label.String = colorLabel;
    caxis([lo hi]);
    xlabel('i (dim 0)'); ylabel('j (dim 1)'); zlabel('k (dim 2)');
    if useOriginalFigureFormat
        FigureFormat(120, 20);
    else
        localFigureFormat(120, 20);
    end
end

function plotRings(data, useOriginalFigureFormat)
% Draws every ring as a filled polygon through its node positions
% (in registered node order, which is already a proper cyclic walk --
% see rings.py), colored by ring size.
    sizes = data.rings.size;
    cmap = jet(max(sizes) - min(sizes) + 1);
    hold on
    for r = 1:numel(data.rings.id)
        rows = data.rings.node_row{r};
        pts = [data.nodes.i(rows), data.nodes.j(rows), data.nodes.k(rows)];
        colorIdx = sizes(r) - min(sizes) + 1;
        patch(pts(:,1), pts(:,2), pts(:,3), cmap(colorIdx,:), ...
            'FaceAlpha', 0.5, 'EdgeColor', [0.3 0.3 0.3]);
    end
    hold off
    colormap(jet(max(sizes) - min(sizes) + 1));
    cb = colorbar; cb.Label.String = 'Ring size (branches)';
    caxis([min(sizes) max(sizes)]);
    xlabel('i (dim 0)'); ylabel('j (dim 1)'); zlabel('k (dim 2)');
    if useOriginalFigureFormat
        FigureFormat(120, 20);
    else
        localFigureFormat(120, 20);
    end
end

function plotChains(data, useOriginalFigureFormat)
% Draws every branch, colored by which chain it belongs to (a distinct
% color per chain, cycling through a qualitative colormap for large
% chain counts).
    nChains = numel(data.chains.id);
    cmap = hsv(max(nChains, 1));
    hold on
    for c = 1:nChains
        branchIds = data.chains.branch_id{c};
        color = cmap(mod(c - 1, size(cmap,1)) + 1, :);
        for bIdRaw = branchIds(:)'
            branchIdx = find(data.branches.id == bIdRaw, 1);
            if isempty(branchIdx)
                continue
            end
            pts = branchCenterline(data, branchIdx);
            plot3(pts(:,1), pts(:,2), pts(:,3), '-', 'Color', color, 'LineWidth', 1.5);
        end
    end
    scatter3(data.nodes.i, data.nodes.j, data.nodes.k, 10, 'k', 'filled');
    hold off
    xlabel('i (dim 0)'); ylabel('j (dim 1)'); zlabel('k (dim 2)');
    title(sprintf('%d chains', nChains));
    if useOriginalFigureFormat
        FigureFormat(120, 20);
    else
        localFigureFormat(120, 20);
    end
end

function plotNetworkWithSTLOverlay(data, resultsFile, useOriginalFigureFormat)
% Loads the companion STL (written next to results.mat by `stereom run
% --export-mat` when the volume came from `stereom extract-regions`),
% converts its vertices from STL FILE units into this graph's
% voxel-index coordinate system, and draws it as a semi-transparent
% surface with the network overlaid on top -- so you can visually check
% that skeleton lines actually run through the interior of the real
% ligaments rather than floating outside them or missing thin features.
%
% Coordinate conversion: voxel = (stl_point - bbox_min_stl_units) /
% pitch_stl_units (see mesh_regions.py's RegionSpec / the
% {name}_stl_alignment.json sidecar it's read from).
%
% Requires MATLAB's built-in `stlread` (R2018b+, returns a
% `triangulation` object). If your MATLAB is older and only has a
% file-exchange `stlread` returning [v,f] or a struct, adjust the
% `tri = stlread(...)` line below to match its signature.

    stlPath = fullfile(fileparts(resultsFile), data.stl_overlay.stl_file);
    if ~isfile(stlPath)
        warning('Companion STL not found at %s -- skipping overlay.', stlPath);
        return
    end

    try
        tri = stlread(stlPath);
        stlVertices = tri.Points;
        stlFaces = tri.ConnectivityList;
    catch ME
        warning(['Could not read %s with the built-in stlread (needs MATLAB ' ...
            'R2018b+). Error: %s'], stlPath, ME.message);
        return
    end

    bboxMin = data.stl_overlay.bbox_min_stl_units;
    pitch = data.stl_overlay.pitch_stl_units;
    voxelVertices = (stlVertices - bboxMin) / pitch;

    hold on
    trisurf(stlFaces, voxelVertices(:,1), voxelVertices(:,2), voxelVertices(:,3), ...
        'FaceColor', [0.75 0.75 0.80], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    light('Position', [1 1 1], 'Style', 'infinite');
    lighting gouraud

    for b = 1:numel(data.branches.id)
        pts = branchCenterline(data, b);
        plot3(pts(:,1), pts(:,2), pts(:,3), '-', 'Color', [0.85 0.10 0.10], 'LineWidth', 1.5);
    end
    scatter3(data.nodes.i, data.nodes.j, data.nodes.k, 15, [0.05 0.05 0.05], 'filled');
    hold off

    xlabel('i (dim 0)'); ylabel('j (dim 1)'); zlabel('k (dim 2)');
    title('Skeleton network (red/black) inside the source STL (grey, transparent)');
    if useOriginalFigureFormat
        FigureFormat(120, 20);
    else
        localFigureFormat(120, 20);
    end
    alpha(0.15); % re-assert in case FigureFormat/localFigureFormat touched material properties
end

function localFigureFormat(azimuth, elevation)
% Minimal built-in equivalent of the original toolbox's FigureFormat.m,
% used when that toolbox isn't on the MATLAB path.
    a = gca;
    box on
    axis image
    a.LineWidth = 1.5;
    view(azimuth, elevation);
    set(gcf, 'color', 'white');
end

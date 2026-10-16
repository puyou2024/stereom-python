function save_all_stereom_figures(resultsFile, outDir)
% =========================================================================
%  save_all_stereom_figures(resultsFile, outDir)
%
%  Non-interactive, batch-mode equivalent of visualize_stereom_results.m:
%  produces every one of that script's sections (network x3 colored by
%  thickness/length/tortuosity, node-degree histogram, length/distance/
%  thickness histograms, orientation polar plot, inter-branch angle
%  histograms, ring size + 3D ring geometry, chains, profile fits, and the
%  skeleton-inside-source-STL overlay) and saves each as a numbered PNG in
%  `outDir`, instead of leaving interactive figure windows open.
%
%  This is the standard post-processing step for every STL run in this
%  project: `outDir` should be a subfolder next to the source STL (e.g.
%  `<stl_folder>/skeleton_results/matlab_figures`), NOT inside
%  stereom_python/.
%
%  Usage (from anywhere, non-interactively):
%    matlab -batch "addpath('path\to\stereom-python\matlab_viz'); save_all_stereom_figures('path\to\results.mat','path\to\out_dir')"
%
%  No analysis is recomputed here -- this only reads what
%  stereom.matlab_export.export_matlab already wrote into results.mat,
%  same separation-of-concerns as the interactive script and as
%  stereom_python's own viz.py.
% =========================================================================

if ~isfolder(outDir)
    mkdir(outDir);
end

data = load(resultsFile);
[~, tag] = fileparts(fileparts(resultsFile));  % parent dir name, for titles

fprintf('Loaded %s\n', resultsFile);
fprintf('  %d nodes, %d branches\n', numel(data.nodes.id), numel(data.branches.id));
if isfield(data, 'rings'); fprintf('  %d rings\n', numel(data.rings.id)); end
if isfield(data, 'chains'); fprintf('  %d chains\n', numel(data.chains.id)); end

n = 0;

%% 1-3: 3D network colored by thickness / length / tortuosity
n = n + 1;
saveNetworkColored(data, data.branches.t_mean_radius_um, 'Mean radius (\mum)', ...
    sprintf('%s: network colored by thickness', tag), fullfile(outDir, sprintf('%02d_network_thickness.png', n)));

n = n + 1;
saveNetworkColored(data, data.branches.length_um, 'Branch length (\mum)', ...
    sprintf('%s: network colored by length', tag), fullfile(outDir, sprintf('%02d_network_length.png', n)));

n = n + 1;
saveNetworkColored(data, data.branches.tortuosity, 'Tortuosity', ...
    sprintf('%s: network colored by tortuosity', tag), fullfile(outDir, sprintf('%02d_network_tortuosity.png', n)));

%% 4: node-type (degree) distribution
n = n + 1;
f = figure('Visible', 'off', 'Position', [100 100 700 500]);
degrees = data.nodes.degree;
edges = (min(degrees) - 0.5):1:(max(degrees) + 0.5);
histogram(degrees, edges, 'Normalization', 'probability', 'FaceColor', [0.20 0.45 0.80]);
xlabel('Node degree (N-type)'); ylabel('Frequency');
title(sprintf('%s: node type distribution (n=%d)', tag, numel(degrees)), 'Interpreter', 'none');
set(gcf, 'color', 'white'); box on
saveas(f, fullfile(outDir, sprintf('%02d_node_degree.png', n)));
close(f);
fprintf('\nNode degree breakdown:\n');
for d = unique(degrees)'
    fprintf('  N-%d: %d (%.1f%%)\n', d, sum(degrees == d), 100 * mean(degrees == d));
end

%% 5: branch length / distance histograms (stairs, overlaid)
n = n + 1;
f = figure('Visible', 'off', 'Position', [100 100 700 500]);
histogram(data.branches.length_um, 30, 'Normalization', 'probability', ...
    'DisplayStyle', 'stairs', 'LineWidth', 1.5, 'EdgeColor', [0.85 0.33 0.10]);
hold on
histogram(data.branches.distance_um, 30, 'Normalization', 'probability', ...
    'DisplayStyle', 'stairs', 'LineWidth', 1.5, 'EdgeColor', [0.20 0.45 0.80]);
legend('curved length l_{ij}', 'node-to-node distance l_{o,ij}');
xlabel('Length (\mum)'); ylabel('Frequency'); box on
title(tag, 'Interpreter', 'none');
set(gcf, 'color', 'white')
saveas(f, fullfile(outDir, sprintf('%02d_length_distance_hist.png', n)));
close(f);

%% 6: branch thickness (radius) histogram
n = n + 1;
f = figure('Visible', 'off', 'Position', [100 100 700 500]);
validThickness = data.branches.t_mean_radius_um(~isnan(data.branches.t_mean_radius_um));
histogram(validThickness, 30, 'Normalization', 'probability', 'FaceColor', [0.30 0.70 0.30]);
xlabel('Mean branch radius (\mum)'); ylabel('Frequency');
title(sprintf('%s: mean radius %.2f \\pm %.2f \\mum (n=%d)', tag, ...
    mean(validThickness), std(validThickness), numel(validThickness)), 'Interpreter', 'none');
set(gcf, 'color', 'white'); box on
saveas(f, fullfile(outDir, sprintf('%02d_thickness_hist.png', n)));
close(f);

%% 7: branch orientation polar plot
n = n + 1;
f = figure('Visible', 'off', 'Position', [100 100 700 700]);
polarplot(deg2rad(data.branches.omega_deg), data.branches.theta_deg, 'o', ...
    'MarkerSize', 4, 'MarkerFaceColor', [0.20 0.45 0.80], 'MarkerEdgeColor', 'none');
rticks(0:30:90); rlim([0 90]);
title(sprintf('%s: orientation (\\theta=radius,\\omega=angle)', tag), 'Interpreter', 'none');
set(gcf, 'color', 'white')
saveas(f, fullfile(outDir, sprintf('%02d_orientation_polar.png', n)));
close(f);

%% 8: inter-branch angle distribution (N-3 and N-4 nodes)
if isfield(data, 'angles')
    n = n + 1;
    nodeDegreeByRow = zeros(numel(data.nodes.row), 1);
    nodeDegreeByRow(data.nodes.row) = data.nodes.degree;
    angleNodeDegree = nodeDegreeByRow(data.angles.node_row);

    f = figure('Visible', 'off', 'Position', [100 100 1100 500]);
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
    sgtitle(tag, 'Interpreter', 'none');
    set(gcf, 'color', 'white')
    saveas(f, fullfile(outDir, sprintf('%02d_interbranch_angles.png', n)));
    close(f);
else
    fprintf('No angles struct in results.mat -- skipping inter-branch-angle figure.\n');
end

%% 9: ring size histogram + 3D ring geometry
if isfield(data, 'rings') && numel(data.rings.id) > 0
    n = n + 1;
    f = figure('Visible', 'off', 'Position', [100 100 700 500]);
    ringSizes = data.rings.size;
    edges = (min(ringSizes) - 0.5):1:(max(ringSizes) + 0.5);
    histogram(ringSizes, edges, 'FaceColor', [0.60 0.40 0.80]);
    xlabel('Ring size (number of branches)'); ylabel('Count');
    title(sprintf('%s: %d rings detected', tag, numel(ringSizes)), 'Interpreter', 'none');
    set(gcf, 'color', 'white'); box on
    saveas(f, fullfile(outDir, sprintf('%02d_ring_size_hist.png', n)));
    close(f);

    n = n + 1;
    f = figure('Visible', 'off', 'Position', [100 100 900 700]);
    saveRings(data);
    title(sprintf('%s: ring geometry (colored by ring size)', tag), 'Interpreter', 'none');
    saveas(f, fullfile(outDir, sprintf('%02d_ring_geometry.png', n)));
    close(f);
else
    fprintf('No rings in results.mat -- skipping ring figures.\n');
end

%% 10: chains
if isfield(data, 'chains') && numel(data.chains.id) > 0
    n = n + 1;
    f = figure('Visible', 'off', 'Position', [100 100 900 700]);
    saveChains(data);
    title(sprintf('%s: %d chains', tag, numel(data.chains.id)), 'Interpreter', 'none');
    saveas(f, fullfile(outDir, sprintf('%02d_chains.png', n)));
    close(f);
else
    fprintf('No chains in results.mat -- skipping chains figure.\n');
end

%% 11: profile fits
if isfield(data, 'profile_fits') && numel(data.profile_fits.branch_id) > 0
    n = n + 1;
    accepted = logical(data.profile_fits.accepted);
    fprintf('\nProfile fits: %d/%d branches accepted for fitting\n', sum(accepted), numel(accepted));

    f = figure('Visible', 'off', 'Position', [100 100 1100 500]);
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
    sgtitle(tag, 'Interpreter', 'none');
    set(gcf, 'color', 'white')
    saveas(f, fullfile(outDir, sprintf('%02d_profile_fits.png', n)));
    close(f);
else
    fprintf('No profile_fits in results.mat -- skipping profile-fit figure.\n');
end

%% 12: skeleton overlaid on the source STL (verification)
if isfield(data, 'stl_overlay')
    n = n + 1;
    f = figure('Visible', 'off', 'Position', [100 100 1000 800]);
    try
        saveSTLOverlay(data, resultsFile);
        title(sprintf('%s: skeleton (red/black) inside source STL (grey)', tag), 'Interpreter', 'none');
        saveas(f, fullfile(outDir, sprintf('%02d_stl_overlay.png', n)));
    catch ME
        fprintf('  STL overlay failed: %s\n', ME.message);
    end
    close(f);
else
    fprintf('No stl_overlay in results.mat -- skipping STL overlay figure.\n');
end

fprintf('\nWrote %d figures to %s\n', n, outDir);

end % save_all_stereom_figures


% =========================================================================
%  Local functions (self-contained -- same logic as
%  ../matlab_viz/visualize_stereom_results.m's local functions, adapted
%  for headless Visible-off figures)
% =========================================================================

function pts = branchCenterline(data, branchIdx)
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

function saveNetworkColored(data, colorValue, colorLabel, titleStr, outFile)
    f = figure('Visible', 'off', 'Position', [100 100 900 700]);
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
    title(titleStr, 'Interpreter', 'none');
    box on; axis image
    view(120, 20);
    set(gcf, 'color', 'white');
    saveas(f, outFile);
    close(f);
end

function saveRings(data)
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
    box on; axis image
    view(120, 20);
    set(gcf, 'color', 'white');
end

function saveChains(data)
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
    box on; axis image
    view(120, 20);
    set(gcf, 'color', 'white');
end

function saveSTLOverlay(data, resultsFile)
    stlPath = fullfile(fileparts(resultsFile), data.stl_overlay.stl_file);
    if ~isfile(stlPath)
        error('Companion STL not found at %s', stlPath);
    end
    tri = stlread(stlPath);
    stlVertices = tri.Points;
    stlFaces = tri.ConnectivityList;

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
    box on; axis image
    view(120, 20);
    set(gcf, 'color', 'white');
    alpha(0.15);
end

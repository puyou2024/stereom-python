# Implementation Specification — Stereom Cellular-Network Analysis Toolbox

**Companion to:** `stereom_network_analysis.md` (concepts, formulas, motivation)
**Subject:** the reference MATLAB implementation accompanying Yang et al. / Chen et al., *Acta Biomaterialia* **107** (2020) 204–217 and 218–231.
**Source analysed:** <https://github.com/Ziling-Wu/Quantitative-3D-structural-analysis-of-the-cellular-microstructures>, branch `master` (64 `.m` files, ~4,400 lines).

---

## 0. How to use this document

This is a **specification of what the code actually does**, derived by reading the source — not a restatement of the papers. Where the code and the papers disagree, the code is documented as the ground truth and the divergence is flagged.

Three intended uses:

1. **Understanding.** §2–§4 describe every stage: exact algorithm, data structures, parameters, and complexity.
2. **Feeding to a coding agent.** §8 is a ported-module map with signatures and dependency order; §10 is a self-contained refactoring brief with acceptance criteria. Those two sections plus §5 (defect register) are what an agent needs.
3. **Debugging your own runs.** §5 and §6 list the defects and convention traps that will bite you, ranked by severity.

> **Read §5 before running anything.** Two functions are missing from the release and one script has a syntax error, so several analysis paths cannot execute as shipped.

---

## 1. What is actually in the repository

### 1.1 Inventory by role

| Path | Lines | Role |
|---|---|---|
| `Code/01_main_dataProcessing.m` | 234 | **Entry point 1** — binary volume → registered graph |
| `Code/02_main_dataAnalysis.m` | 132 | **Entry point 2** — dispatcher that `open`s each analysis script |
| **Skeletonization** | | |
| `toolbox/skeleton/Skeleton3D.m` | 151 | Lee–Kashyap–Chu parallel medial-axis thinning (3rd party, see §1.2) |
| `toolbox/skeleton/FillEulerLUT.m`, `p_EulerInv.m`, `p_is_simple.m`, `p_oct_label.m`, `pk_get_nh*.m` | 723 | Euler-characteristic lookup tables and simple-point tests for the above |
| **Network registration** | | |
| `node_identification.m` | 15 | classify skeleton voxels as nodes |
| `branch_trim.m` | 41 | one pass of dangling-branch removal |
| `branch_sort2.m` | 76 | build `bran` / `connect` / `node` from a trimmed skeleton |
| `merge_connectednodes.m` | 72 | merge directly adjacent nodes |
| `merge_double.m` | 94 | **dissolve N-2 nodes** (splice branch pairs) |
| `merge_nearby.m` | 121 | merge node clusters within `d_m` |
| **Geometry / descriptors** | | |
| `branch_order.m` | 58 | order branch voxels into a path |
| `branch_length.m` | 19 | curved length + node-to-node distance |
| `branch_thickness.m` | 40 | quick mid-branch radius (cube growing) |
| `branch_thick.m` | 45 | alternative cross-section area (**broken**, §5) |
| `bran_direction.m` | 7 | unit direction from endpoint pairs |
| `interangle.m` | 8 | angle between two chord vectors (**radians**) |
| `myangle.m`, `myanglez.m`, `vectorAngle.m` | 78 | orientation angle pairs |
| `ori3.m` | 29 | N-3 node normal + planarity angle |
| `offangle.m` | 11 | legacy planarity variant (**incorrect**, §5) |
| `findloop.m` (General) | 309 | ring detection, unrolled BFS to depth 7 |
| `findloop.m` (network_tools) | 193 | older/shorter variant of the same |
| `ringSize_4E/5E/6E/7E.m` | 82 | polygon area by Heron fan triangulation |
| `planeFitting.m` | 8 | ordinary least squares plane (`z` on `x,y`) |
| `planeFitting_v2.m` | 23 | **total least squares plane via SVD** — the one actually used |
| `chain.m` | 23 | recursive chain growth (**missing dependency**, §5) |
| `DeleteRepeatedBran.m` | 24 | remove parallel edges |
| `neighbor.m`, `vecdist.m`, `mydistance.m`, `findbran.m`, `findNearNode.m`, `survalue.m` | ~110 | primitives |
| **Analysis drivers** | | |
| `main_node.m`, `main_length.m`, `main_thickness.m`, `main_branchfitting.m`, `main_profile.m`, `main_branchOrientation.m`, `main_interBranchAngle.m`, `main_correlation.m`, `main_ring.m`, `main_ring_plot.m`, `main_chain.m`, `main_fft.m` | ~2,150 | per-descriptor scripts, mixed computation and plotting |
| **Plotting** | `FigureFormat.m`, `PlotScatter*.m`, `PlotBox.m`, `PlotSurf.m`, `isboundary.m`, `shrink.m`, `histofrequency.m` | ~190 | visualization + boundary test |

Demo data: `Code/Data/Binary.zip` — 308 binary TIFF slices, volume 386 × 386 × 308 at 0.65 µm/voxel (the Paper I representative volume).

### 1.2 Provenance of the skeletonizer — an important simplification

The papers describe "a custom-written 3D thinning algorithm based on an iterative erosion method." The code tells a different story: `Skeleton3D.m` is Philip Kollmannsberger's MATLAB File Exchange implementation of

> Lee, Kashyap & Chu, *Building skeleton models via 3-D medial surface/axis thinning algorithms*, CVGIP **56**(6):462–478, 1994.

**This matters enormously for porting.** `skimage.morphology.skeletonize(vol, method='lee')` (formerly `skeletonize_3d`) implements the *same* published algorithm. You can substitute it and expect voxel-identical or near-identical skeletons — no reimplementation needed. That removes ~875 lines (20% of the repo) from any porting effort.

The only behavioural detail to preserve: `Skeleton3D` pads the volume with one voxel of zeros before thinning to avoid edge effects, and accepts an optional `spare` mask of voxels that must be preserved.

### 1.3 Paper ↔ code coverage gaps

| Described in the papers | Present in this release? |
|---|---|
| Skeletonization, trimming, node merging | **Yes**, complete |
| Node types, inter-branch angles, planarity | **Yes** |
| Branch length, distance, tortuosity, thickness, orientation | **Yes** |
| Branch profile fitting | **Yes**, but a *pure quadratic with no linear term* (§3.12) |
| Ring detection + ring size + ring orientation | **Yes** |
| Ring flatness filter ($R^2 \ge 0.5$) | **No** — present only as commented-out code in `findloop.m` |
| Redundancy filter (higher-B composed of lower-B) | **No** |
| **Chain alignment factor (CAF) formula** | **No** — `chain.m` uses a threshold rule instead (§3.17) |
| 3D-FFT of node array | **Yes** |
| Even fourth-order profile fit (Paper II Eq. 1) | **No** — Paper II's descriptors are not in this repo |
| Beam-model export / FE deck generation | **No** — the Python exporter and Abaqus decks are not released |
| Hierarchical CNN reconstruction / segmentation | **No** — separate work, not in this repo |

**Conclusion:** this repository implements **Paper I** (methodology on a small volume). Paper II's additions — the quartic profile model, the plane-fit flatness filter, the septum analysis, and the FE pipeline — are **not** included and must be written from the paper descriptions.

---

## 2. Canonical data model

### 2.1 The four variables everything revolves around

Every function in the toolbox operates on some subset of these. Understanding them is 80% of understanding the code.

| Variable | MATLAB type | Meaning |
|---|---|---|
| `orig` | `double` array `[nx, ny, nz]` | binary volume, rescaled to `[0,1]` |
| `datas` | `1×3 double` | `size(orig)` — passed to nearly every function so linear indices can be decoded |
| `skel_c` / `trimskel` | logical/double array, same size | skeleton volume, 1 on centerline voxels |
| `node` | `N×2 double` | column 1 = **linear index** of the node voxel; column 2 = **node type** |
| `bran` | `1×M cell` | `bran{i}` = column vector of **linear indices** of the voxels of branch *i* |
| `connect` | `M×2 double` | `connect(i,:)` = linear indices of the two nodes bounding branch *i* |

**The whole graph is expressed in linear voxel indices.** There are no separate node IDs; a node *is* its voxel index. This is compact but has three consequences you must respect when refactoring:

- Node identity changes whenever a node moves (merging relocates a node to a cluster centroid → new index → every `connect` entry referencing it must be rewritten). The code does exactly this, by search-and-replace over `connect`.
- `datas` must accompany every index. Lose it and the indices are meaningless.
- Index magnitudes are large: for the demo volume, up to 386·386·308 ≈ 4.6 × 10⁷. §5 shows where this overflows a numerical hash.

### 2.2 `node(:,2)` changes meaning mid-pipeline — a trap

This is the single most confusing thing in the codebase.

| Produced by | `node(:,2)` means |
|---|---|
| `node_identification.m` | **count of skeleton voxels in the 26-neighbourhood** (a voxel-level proxy for degree) |
| `branch_sort2.m`, `merge_*.m` | **true graph degree** — number of rows of `connect` referencing this node |

A junction whose skeleton is locally fat can have 5 or 6 neighbouring voxels while joining only 3 branches. The first number is a *heuristic used to find candidate junctions*; only the second is the coordination number reported in the papers. **Never report node-type statistics from a pre-`branch_sort2` node table.**

*Refactor note:* give these two fields different names (`voxel_degree` and `degree`). Do not overload one column.

### 2.3 Axis convention — implicit and hardcoded

MATLAB's `ind2sub(datas, idx)` returns `[d1, d2, d3]` = (row, column, slice). The orientation functions then hardcode:

| Array dimension | Physical axis | Role |
|---|---|---|
| **dim 1** (rows) | **L** — longitudinal | polar axis; θ is measured from it |
| **dim 2** (columns) | **C** — circumferential | — |
| **dim 3** (slices) | **R** — radial | in-plane reference; ω is measured from it |

Evidence: `myangle.m` computes the polar angle as `atand( sqrt(Δd3² + Δd2²) / |Δd1| )` — polar axis is dim 1. Its azimuthal angle is `atan(|Δd2| / |Δd3|)`, zero when Δd2 = 0 — reference is dim 3. And `merge_nearby.m` varies its merging threshold as a function of the **dim-3** coordinate, which is the graded direction (radial) in the specimen.

> **Consequence:** you must load your volume so that slice index = radial/gradient direction and row index = the axis you want θ measured from. Otherwise every orientation result is silently wrong. This is *not* documented anywhere in the repo.

### 2.4 Angle-name collision between code and papers

| Papers | `myangle.m` / `vectorAngle.m` | Range | Meaning |
|---|---|---|---|
| **θ** (misorientation) | returns as **`phi`** | 0°–90° | angle from the L axis |
| **ω** (in-plane rotation) | returns as **`theta`** | 0°–360° | azimuth in the C–R plane, from R |

The names are **swapped**. `main_branchOrientation.m` stores them as `branOrien = [theta, phi]`, so **column 1 is ω and column 2 is θ**. §5 documents a live bug caused by exactly this.

### 2.5 Proposed schema for a refactor

Replace the four loose variables with explicit tables. This is what a ported implementation should expose.

```
Volume:
    data        : bool[nx, ny, nz]
    voxel_size  : float            # µm/voxel, isotropic
    axes        : ('L','C','R')    # dim -> physical axis, EXPLICIT

Nodes (table, one row per node):
    id          : int              # stable, independent of position
    voxel       : int[3]           # (i,j,k), not a linear index
    degree      : int              # true graph degree
    voxel_degree: int              # skeleton-neighbour count (diagnostic only)

Branches (table, one row per branch):
    id          : int
    node_a, node_b : int           # node ids
    path        : int[n,3]         # ORDERED centerline voxels
    length      : float            # µm, curved
    distance    : float            # µm, Euclidean node-to-node
    tortuosity  : float
    profile     : float[n]         # radius, µm, sampled along path
    t_min, t_cen, t_mean : float
    theta, omega: float            # degrees, paper convention

Rings (table):
    id, n_branches, node_ids[], branch_ids[]
    centroid[3], normal[3], area, d_ring, flatness
```

Keying on stable integer IDs rather than voxel indices removes the rewrite-on-merge fragility entirely, and makes the graph serializable to any standard format (GraphML, HDF5, Parquet).

---

## 3. Stage-by-stage implementation

Each subsection gives: **code path → exact algorithm → parameters → complexity → failure modes → refactor notes.**

### 3.1 Volume I/O

**Code path:** `01_main_dataProcessing.m` §1.

Reads a numbered TIFF stack via a filename template (`imageName` + zero-padded index + extension), assembles a 3D array slice by slice, then rescales to `[0,1]` by min–max.

**Parameters:** `startPage`, `endPage`, `sigDigits`, `style`, `imageName`, `folder`.

**Failure modes:**
- The array is grown inside the loop without preallocation — O(n²) memory traffic on large stacks.
- Min–max rescaling of an already-binary stack is a no-op unless a slice is constant, in which case it divides by zero → `NaN` slice. Guard this.
- `double` storage: the 386³ demo volume is ~370 MB as double; a 0.32 mm³ Paper II volume would be hundreds of GB. **Use `logical`** — 8× smaller than `uint8`, 64× smaller than `double`.

**Refactor:** preallocate; store `bool`; read with `tifffile`/`imageio`; support chunked/lazy loading (`dask`, `zarr`) for large volumes.

### 3.2 Skeletonization

**Code path:** `Skeleton3D(orig)`.

Lee–Kashyap–Chu parallel medial-axis thinning: iteratively delete "simple" border points (those whose removal preserves topology, tested via an Euler-characteristic lookup table and octant labelling) in six directional sub-iterations, until no point can be removed.

**Complexity:** O(iterations × N_solid_voxels), with the LUT making the simple-point test O(1).

**Failure mode:** produces medial *surfaces* for plate-like objects, not lines. The downstream graph code assumes lines. Verify on your material.

**Refactor:** replace with `skimage.morphology.skeletonize(vol, method='lee')`. Keep the zero-padding behaviour.

### 3.3 Node identification

**Code path:** `node_identification.m`.

```
for each skeleton voxel v:
    n  = 27-neighbourhood of v (clipped at volume borders, INCLUDES v)
    k  = count of skeleton voxels in n            # includes v itself
    if k >= 4:                                     # i.e. >= 3 true neighbours
        record node (index = v, voxel_degree = k - 1)
```

**Parameters:** the threshold `k >= 4`, i.e. the paper's "more than two neighbouring voxels."

**Complexity:** O(N_skel × 27). `neighbor.m` rebuilds a `meshgrid` and calls `sub2ind` in a scalar loop for every voxel — this is the dominant cost of the stage and is trivially vectorizable.

**Failure modes:**
- Border clipping means a voxel on the volume face has fewer than 26 neighbours and can be missed. Pad first.
- A thick junction yields a *cluster* of qualifying voxels, not one node. Fixed later by merging — by design.
- A perfectly straight diagonal chain can have 2 neighbours; a corner can momentarily have 3. Expect false positives; that is what trimming and merging clean up.

**Refactor:** one convolution. `k = convolve(skel.astype(uint8), ones((3,3,3)))`; `nodes = skel & (k >= 4)`. Two lines, ~1000× faster.

### 3.4 Trimming loop

**Code path:** `01_main_dataProcessing.m` §3 Step 1, calling `branch_trim.m`.

**Outer loop (in the driver):**

```
nodes = node_identification(skel)
repeat:
    n_before = count(nodes)
    skel     = branch_trim(nodes, skel)
    nodes    = node_identification(skel)
until count(nodes) == n_before
```

Convergence is tested on **node count**, not on branch count or on an explicit "no dangling branches" predicate.

**`branch_trim` — one pass:**

```
1. Zero out all node voxels in the skeleton.
2. Label the remainder with 26-connectivity  ->  candidate branch segments.
3. For each segment S:
     endpoints = { v in S : v has exactly 1 neighbour inside S }
     valid_ends = count of endpoints adjacent to EXACTLY ONE node voxel
     keep S  iff  valid_ends == 2
     special case |S| == 1: keep iff that voxel touches exactly 2 nodes
4. Restore node voxels.
```

So a segment survives only if **both ends attach to a node**. Dangling branches (one free end) are deleted. Note this also silently deletes segments whose endpoint touches *two or more* nodes — such segments are dropped rather than resolved.

**Complexity:** O(passes × N_skel × 27). Each pass re-runs `bwlabeln` over the whole volume.

**Failure modes:**
- The counter starts at 1 and increments, so the magic value `3` means "two valid ends." Opaque but correct.
- A branch forming a closed loop with no endpoint has `valid_ends == 0` and is deleted. Small isolated rings are lost.
- Convergence on node count can in principle oscillate; there is no iteration cap.

**Refactor:** operate on the graph, not the volume. After one graph build, repeatedly remove degree-1 nodes and their incident edges until none remain — this is `networkx`-level work and is O(V+E) total instead of O(passes × volume).

### 3.5 Branch/node registration

**Code path:** `branch_sort2.m`.

Same skeleton decomposition as `branch_trim`, but now *recording* rather than filtering:

```
1. Zero node voxels; label remainder (26-connectivity).
2. For each segment:
     find its endpoints (1 neighbour within the segment)
     for each endpoint:
         adjacent_nodes = node voxels in its 27-neighbourhood
         if |adjacent_nodes| == 1: register that node
         if |adjacent_nodes| >  1: register the ROUNDED CENTROID of them
                                   as a new node position     <-- inline merge
     keep the segment iff exactly 2 ends were registered
3. Single-voxel segments touching exactly 2 nodes are kept.
4. Delete self-loops:  rows where connect(:,1) == connect(:,2).
5. Rebuild the node list as unique(connect(:)); set
   node(:,2) = number of occurrences in connect   <-- TRUE DEGREE from here on
```

**Two important behaviours:**
- Step 2's centroid substitution is an **undocumented merging step** that happens before the explicit merge functions. It can invent a node position that is not a skeleton voxel.
- Step 5 is where `node(:,2)` switches meaning (§2.2).

**Failure modes:**
- `connect` is written to before it is preallocated or guaranteed non-empty; if the first segments all fail the tests, MATLAB errors or silently creates a mis-shaped array.
- Segments with >2 valid ends print a bare `'3'` to the console and are dropped without record.
- **Parallel edges are preserved** here (two branches joining the same node pair). They are removed much later, in `main_length.m`, and only there.

### 3.6 Node merging — three distinct functions

The driver calls them in a fixed order that matters:

```
[c2,n2,b2] = merge_connectednodes(connect, node, bran, datas)
[c3,n3,b3] = merge_double(c2,n2,b2, datas, 5)     % N-2 dissolution, guarded
[c4,n4,b4] = merge_nearby(c3,n3,b3, 10, 1, datas) % d_m clustering
[c5,n5,b5] = merge_double(c4,n4,b4, datas)        % N-2 dissolution, unguarded
```

#### 3.6.1 `merge_connectednodes` — adjacent nodes

Merges nodes that are **direct 26-neighbours** of each other.

```
for each node v (in ascending linear-index order):
    if v already absorbed: skip
    candidates = the next 500 nodes in the list          <-- WINDOW
    cluster = { v } U { candidates that are 26-neighbours of v }
    if |cluster| > 1:
        survivor = MAX linear index in the cluster       <-- not the centroid
        rewrite every connect entry referencing a cluster member -> survivor
        append the absorbed node's voxel into the branch path (to keep paths contiguous)
delete self-loops; rebuild node list and degrees
```

**Two divergences from the papers:**
- The paper says merged nodes are placed at the **cluster centroid**. This function uses the **maximum linear index** instead — an arbitrary member, biased toward higher slice indices.
- The 500-node lookahead window is a performance hack that assumes nodes are sorted by linear index so that spatial neighbours are nearby in the list. For a volume of cross-section `nx·ny`, two voxels adjacent in dim 3 differ in index by `nx·ny` — for the demo volume, 148,996. **A 500-entry window cannot see them unless the node list happens to be dense.** Adjacent-node pairs separated along dim 3 will be missed.

#### 3.6.2 `merge_double` — N-2 node dissolution

This is the paper's "replace N-2 nodes with branch points" step.

```
for each node v:
    if degree(v) == 0: drop
    if degree(v) == 1 or > 2: keep as-is
    if degree(v) == 2:
        e1, e2 = the two incident branches
        if called with 5 arguments and min(len(e1), len(e2)) > constant:
            keep v (do not merge)                        <-- length guard
        else:
            merge: connect(e1) := the two far-end nodes
                   splice paths: choose the concatenation order by comparing
                   endpoint distances, inserting v's voxel between the two paths
                   delete e2
delete self-loops; rebuild
```

**The `constant` guard.** Called as `merge_double(..., 5)` the first time: an N-2 node is dissolved only if the **shorter** of its two branches is ≤ 5 voxels. This prevents merging two genuinely long branches into one very long one. Called with 4 arguments the second time, the guard is disabled and *all* remaining N-2 nodes are dissolved.

*(The source comment records that this was changed from `max(len)` to `min(len)` — the fix makes the guard conservative.)*

**Splice-order logic:** compares `dist(path1.first, path2.last)` against `dist(path1.last, path2.first)` and concatenates whichever is smaller, so the merged path stays geometrically ordered. Necessary because branch paths have no intrinsic direction.

#### 3.6.3 `merge_nearby` — the $d_m$ cluster merge

This implements the paper's main merging step.

```
for each node v (ascending index order):
    if v already absorbed: skip
    if open_flag:                                  <-- POSITION-DEPENDENT d_m
        r = dim-3 coordinate of v
        d_m = 12 if r < 1700
              10 if 1700 <= r < 2200
               8 otherwise
    candidates = the next 1000 nodes                <-- WINDOW
    cluster = { v } U { c in candidates : euclidean_dist(c, v) <= d_m }
    if |cluster| >= 2:
        delete every branch whose two endpoints are both inside the cluster
        centre = ROUNDED CENTROID of the cluster voxels
        rewrite all connect entries referencing cluster members -> centre
delete self-loops; rebuild node list and degrees
```

**Parameters:** `merge_thres` (voxels) and `open` (0/1).

> ### ⚠ The `open` flag overrides `merge_thres` entirely
>
> The driver calls `merge_nearby(c3, n3, b3, **10**, **1**, datas)`. Because `open == 1`, the hardcoded ladder replaces the passed value on every iteration. **The `10` has no effect.** For the demo volume (308 slices), `r < 1700` is always true, so the effective threshold is **12 voxels = 7.8 µm**, not the 6.5 µm reported in Paper I.
>
> These three magic numbers are specimen-specific radial positions in voxels, left over from the Paper II gradient specimen. **Delete this branch and pass `d_m` explicitly** — as a scalar, or as a callable of position for graded materials.

**Correct behaviours worth preserving:** the centroid placement (matches the paper, unlike `merge_connectednodes`), and the deletion of intra-cluster branches before rewiring (otherwise they become self-loops).

**Failure mode:** the same 1000-node window problem as §3.6.1, and the merge is **order-dependent** — clusters are grown greedily from the lowest-index member, so results depend on node ordering.

**Refactor:** build a KD-tree over node coordinates, query pairs within `d_m`, form a proximity graph, take its connected components (union–find) as clusters, and place each survivor at its centroid. This is order-independent, window-free, and O(N log N).

### 3.7 Boundary exclusion

**Code path:** `isboundary.m`, invoked from `main_node.m` and `main_length.m`.

```
boundaryLimit = 30    % voxels; set for the 386x386x308 demo volume
```

A node or branch is excluded if any of its voxels lies within `boundaryLimit` of any face. Note this is applied **in the analysis scripts, not during registration**, so the saved graph still contains boundary elements — filtering is a per-analysis decision. Keep it that way; it is the right design.

At 0.65 µm/voxel, 30 voxels = 19.5 µm — matching Paper II's stated exclusion depth, and ≈ one mean branch length.

---
### 3.8 Branch path ordering

**Code path:** `branch_order.m`. **Must run before any length or profile computation** — the raw `bran{i}` from `bwlabeln` is an unordered voxel set.

```
for each branch i:
    voxels = unique(bran{i})
    start  = the voxel of `voxels` adjacent to connect(i,1)
             (fallback: the voxel nearest to connect(i,1))
    path   = [start]
    while |path| < |voxels|:
        next = a neighbour of path.last that is in `voxels` and not yet in `path`
        if none:                                   # the chain broke (gap)
            next = nearest unvisited voxel         # JUMP
        append next
    # trailing-voxel repair
    if dist(path[-1], path[-2]) > 10 voxels:
        if dist(path[-1], path[0]) <= 10: move path[-1] to the front
        else:                              drop path[-1]
```

**Why the jump exists:** merged branches (from `merge_double`) are concatenations of two paths plus an intervening node voxel, and can contain small gaps. The nearest-unvisited jump keeps the walk going.

**Failure modes:**
- The jump silently accepts arbitrarily large gaps and inflates the accumulated length. **Log every jump and its magnitude**; a jump larger than ~2 voxels means something upstream is wrong.
- The magic `10` (voxels) in the repair step is unscaled — it is a raw voxel count, not a physical distance.
- A branch that self-intersects or forks (should not happen post-cleaning, but does) will be linearized arbitrarily.

**Refactor:** order by graph traversal on the branch's own voxel adjacency graph, or better, retain the ordered path directly from a skeleton-to-graph library (`skan` returns ordered paths natively). Then this whole function disappears.

### 3.9 Length, distance, tortuosity

**Code path:** `branch_length.m`.

```
curved_length(i) = sum over consecutive path voxels of euclidean_dist
                 + dist(connect(i,1), nearest end of the path)
                 + dist(connect(i,2), nearest end of the path)

distance(i)      = dist(connect(i,1), connect(i,2))

tortuosity(i)    = curved_length(i) / distance(i)
```

The two "stub" terms matter: node voxels are excluded from `bran{i}` (they were zeroed before labelling), so without them every branch would be short by roughly two voxels at each end.

All outputs are **in voxels**; the driver multiplies by `pixelSize` afterwards.

**Failure modes:** if the path is unordered, the consecutive-distance sum is meaningless (a random-walk length, wildly inflated). This is the hard dependency on §3.8. Assert `tortuosity >= 1` on every branch.

### 3.10 Parallel-edge removal

**Code path:** `DeleteRepeatedBran.m`, called once from `main_length.m` after a first length pass.

```
for each branch, find all branches connecting the same unordered node pair
if more than one exists: keep the one with the SMALLER tortuosity, delete the rest
```

The rationale in the source comment: duplicate branches arise from small holes *inside* a strut, which split one thick branch into two thin curved ones. Keeping the straighter one is the right choice.

**Note the ordering dependency:** tortuosity must be computed before deduplication, and lengths must then be **recomputed** afterwards. `main_length.m` does exactly this — two passes of `branch_length`. Preserve that.

### 3.11 Thickness extraction

This is the most involved and most expensive stage, and the implementation differs materially from the papers' description.

#### Stage A — surface extraction

```
cont = edge3(orig, 'approxcanny', 0.6)
```

A 3D Canny edge map of the binary volume with threshold 0.6 → the set of **surface voxels**. Everything downstream measures distances to `cont`.

#### Stage B — rough per-branch scale (`main_thickness.m` §1)

```
for each branch i:
    v = FIRST voxel of the branch path
    w = 1
    repeat: w = w + 1
    until the cube [v-w, v+w]^3 of `cont` contains >= 5 edge voxels
    rough(i) = w
```

A cube-growing (Chebyshev/L∞) search for the nearest surface. `rough(i)` is used **only** as an outlier guard in Stage C, never reported.

> Note `branch_thickness.m` is a near-duplicate that uses the branch **midpoint** and a threshold of **1** edge voxel instead of 5. It is not called by `main_thickness.m`. Two implementations of the same idea with different constants — pick one when refactoring.

#### Stage C — full radial profile (`main_thickness.m` §2)

```
for each branch i:
    if |path| < 10 voxels: SKIP the branch entirely
    r1 = chord vector from path[1] to path[end]        # branch axis proxy
    for j in 2 .. |path|-1:                            # interior voxels only
        p = path[j]
        S = { surface voxels within a sphere of radius 20 voxels of p }
        for each s in S:
            c = s - p
            angle = |acos( (c . r1) / (|c| |r1|) )| in degrees  - 90
        keep those with |angle| < 2 degrees        # within 2 deg of the plane
                                                   # perpendicular to the axis
        if none: skip this sample point
        d = euclidean distances from p to the kept surface voxels
        if any d < rough(i) + 11:   profile(j) = mean of those d
        else:                       profile(j) = mean of all d
```

**What this actually computes:** the **mean distance from the centerline point to the surface, over surface voxels lying within ±2° of the perpendicular plane** — i.e. a mean cross-sectional radius.

This is *not* the "radius of an equivalent circle with the same area as the intersecting profile" described in the papers. For a circular section the two agree; for a non-circular section they differ (mean radius vs. area-equivalent radius). Since the papers themselves note stereom sections are slightly non-circular, this is a real, if small, divergence.

#### Stage D — reported thickness

```
branthickness(i) = min( profile ) * pixelSize
```

**The reported "central branch thickness" is the MINIMUM radius along the branch**, in µm — i.e. it is $t_{\min}$ as a **radius**, not a diameter, and not a central value despite the variable name.

> ### ⚠ Radius vs. diameter
> Paper I reports "mean branch thickness 6.2 ± 1.5 µm" and separately notes the merging distance was set to "roughly the mean branch radius (6.2 µm)." Both cannot be true unless "thickness" here means **radius**. The code confirms it: `branthickness` is a distance from centerline to surface. **Treat all thickness values in this pipeline as radii** unless you verify otherwise, and check the factor of two before comparing with any external dataset.

**Failure modes, in severity order:**

1. **Performance catastrophe.** `tempsphere = cont .* (sphere mask over the FULL volume)` allocates and multiplies an array the size of the entire volume **for every interior voxel of every branch**. For the demo volume that is ~4.6 × 10⁷ elements × ~30,000 sample points ≈ 10¹² operations. This is why the script tells you to "grab a cup of coffee." Fix by cropping a local window, or by precomputing a KD-tree of surface voxels once.
2. **No bounds checking** in the cube-growing loops: `cont(x0-w:x0+w, ...)` throws an index error for any branch endpoint within `w` of a volume face. Pad the volume or clamp.
3. **Chord as axis proxy.** `r1` is the straight endpoint-to-endpoint vector, so for a curved branch the "perpendicular" plane is misaligned near the ends, inflating the measured radius there. Use a **local tangent** (finite difference over a few path voxels, or a smoothed spline derivative) instead.
4. **Magic constants:** sphere radius 20 voxels, angular tolerance 2°, outlier guard `rough + 11`. All unscaled voxel/degree literals with no derivation. The 2° tolerance in particular controls how many surface voxels are averaged — too tight and the sample is empty (point silently skipped), too loose and you average over an oblique band.
5. **Branches shorter than 10 voxels get no profile at all** and are absent from `allthickness`. In the dense edge region, where struts are short and fat, this could exclude a large fraction. Report the exclusion count.

**Refactor (strongly recommended):** replace Stages A–D with a **Euclidean distance transform**:

```
edt = distance_transform_edt(binary_volume)      # µm, once, for the whole volume
profile(j) = edt[path[j]]                        # inscribed radius at each point
```

This is the standard "local thickness" definition, is O(N) for the whole volume, requires no magic constants, and needs no surface extraction. It gives the **inscribed** radius rather than the mean radius — slightly different but far better defined, and directly comparable to `porespy.filters.local_thickness` and BoneJ. If you need the paper's equal-area definition, extract a true cross-section on the local tangent frame and compute `r = sqrt(area/π)`.

### 3.12 Profile fitting

**Code path:** `main_branchfitting.m` (buggy) and `General/main_profile.m` (corrected duplicate).

```
arclength(1) = 0
arclength(j) = cumulative euclidean distance along the path, x pixelSize
a = argmin( profile )                              # index of the thinnest point
z = arclength - arclength(a)                       # coordinate ANCHORED AT THE WAIST
t = profile * pixelSize

REJECT the branch if:
    any NaN in t
    |t| < 2 samples
    max(|z|) > 25 µm   OR   total length > 35 µm   OR   total length < 2 µm

fit:  t(z) = a0 + b*z^2        # fittype({'1','x^2'}) — NO linear term
collect: b   (the curvature coefficient)

normalized outputs, pooled across all branches:
    z / total_length        vs.      t / t_min
```

**Three things this settles about the papers:**

1. **The coordinate is anchored at the thickness minimum**, not the branch midpoint. That is Paper **II**'s convention (Eq. 1), already present in the Paper I code.
2. **Normalization is by `t_min`**, not by midpoint thickness — again Paper II's convention.
3. **The factor-of-two ambiguity in Paper I's Eq. (3) is resolved.** The normalized abscissa is `z / total_length` where `z` is measured from the waist. Because the waist can sit anywhere along the branch, `z/L` spans roughly `[-1, +1]` — which is exactly the plotted range. **It is the full length, not the half length.** (This corrects the caveat raised in §12.3 of the companion document.)

**The fitted model has no linear term**, unlike published Eq. (3). Consistent with the reported linear coefficient of −0.026 ≈ 0, but it means the code cannot reproduce that published number — it was obtained from a different fit. If you want Paper I's Eq. (3), add `'x'` to the `fittype` basis. If you want Paper II's quartic, you must write it yourself:

```
fit  t/t_min = (a * ((x - x_min)/L)^2 + 1)^2      # one parameter, a
```

**Filter values differ from the paper.** The paper states "tortuosity < 2 and length within 5–40 µm (98.6% of branches)." The code uses **length in (2, 35) µm** and **max |z| ≤ 25 µm**, with no tortuosity test at this stage. Reconcile deliberately; do not assume the paper's numbers are in force.

### 3.13 Orientation

**Code path:** `myangle.m` (from a node pair), `vectorAngle.m` (from a direction vector), driver `main_branchOrientation.m`.

```
phi   = atand( sqrt(d2^2 + d3^2) / |d1| )       # 0..90   -> paper's THETA
theta = atand( |d2| / |d3| ), quadrant-resolved  # 0..360  -> paper's OMEGA
```

Quadrant resolution: the vector is first given a canonical sign (flipped so that `d1 > 0`), then the raw arctangent is mapped into one of four quadrants by the signs of `d2` and `d3`. This is a hand-rolled `atan2` — replace it with `atan2` and a modulo, which is both shorter and numerically safer.

Branch orientation is computed from the **node-to-node chord**, not from the local tangent at either end. With tortuosity ≈ 1.2, chord and tangent differ measurably; the choice is defensible (it is the branch's net direction) but must be stated.

**Degenerate cases:** `d1 == 0` returns `phi = 0` — **wrong**; a branch exactly perpendicular to the L axis has θ = 90°, not 0°. Similarly `d3 == 0` returns `theta = 0`. Both are division-guard shortcuts that produce silently incorrect angles for axis-aligned branches. Since axis-aligned branches are *exactly* the population of interest in a textured material, **this is a substantive defect** (§5).

### 3.14 Inter-branch angles

**Code path:** `interangle.m`, driver `main_interBranchAngle.m`.

```
interangle(n0, n1, n2):
    v1 = pos(n1) - pos(n0)
    v2 = pos(n2) - pos(n0)
    return acos( (v1 . v2) / (|v1| |v2|) )        # RADIANS
```

Angles are computed between **node-to-node chord vectors**, not local tangents at the junction. For curved branches these differ; with mean tortuosity 1.2 the discrepancy is a few degrees. The papers do not state which is used. Document your choice.

**Driver logic:**

```
for each node i:
    neighbours = all nodes joined to i by a branch
    for j in 1..len(neighbours):
        angle_c(i,j) = interangle( i, neighbours[j], neighbours[(j mod n)+1] )
```

> ### ⚠ Only cyclically adjacent pairs are measured
>
> For an **N-3** node this yields 3 angles = all $\binom{3}{2}$ pairs. Correct.
>
> For an **N-4** node this yields **4** angles, not the **6** distinct pairs. Which 4 depends on the arbitrary order in which `find` returned the neighbours. So the reported N-4 mean, min and max are computed over an order-dependent subset. The published N-4 statistics are affected by this.
>
> **Fix:** enumerate `itertools.combinations(neighbours, 2)` — $n(n-1)/2$ angles, order-independent.

**Node typing in the driver** is inferred from zero-padding of the `angle_c` array (`col4 == 0 & col3 ≠ 0` → N-3). Fragile; use the degree column instead.

**Numerical guard:** the driver wraps results in `abs()` and `real()` because `acos` returns a complex number when floating-point error pushes its argument past ±1. **The correct fix is to clamp:** `acos(clip(x, -1, 1))`. Taking `real()` of a complex angle silently maps it to 0° or 180°.

### 3.15 N-3 node orientation and planarity

**Code path:** `ori3.m`.

```
ori3(n0, n1, n2, n3):
    v1 = pos(n2) - pos(n1);  v2 = pos(n3) - pos(n1)      # both normalized
    normal = normalize( cross(v2, v1) )                   # normal to the plane
                                                          # through the 3 NEIGHBOURS
    u_k = normalize( pos(n_k) - pos(n0) )  for k = 1,2,3
    theta_k = acosd( u_k . normal )
    if mean(theta) > 90: normal = -normal; out = 180 - max(theta)
    else:                                  out =       max(theta)
    return normal, out
```

Two notes:

- The plane is fitted through the **three neighbour nodes**, and the node orientation is that plane's normal. The companion document's closed form for the *equiangular central axis* — `v ∝ (u1−u2) × (u2−u3)` — is the alternative reading of the papers' definition. The two coincide only when the three branches have equal length. **They are different estimators.** `ori3`'s version is cheaper and more stable; the equiangular version is what the papers' text literally describes. Choose one and state it.
- `out` is `max(theta_k)`, an **angle in degrees**, whereas the paper's planarity index is a **cosine in [0,1]**. Convert with `p = |cos(out°)|`, and verify the sense: perfectly planar ⇒ each branch is at 90° to the normal ⇒ `out = 90°` ⇒ `p = 0`. ✔

`offangle.m` is a legacy variant that normalizes `v1` and `v2` before crossing but then divides by `|v3|` only — the cross product of two unit vectors is **not** a unit vector (its norm is the sine of their included angle), so the `acos` argument is not a cosine. **This function is mathematically incorrect; do not port it.**

### 3.16 Ring detection

**Code path:** `General/findloop.m` (309 lines), driver `main_ring.m`. A shorter, older variant exists at `network_tools/findloop.m`.

**Algorithm:** a **manually unrolled breadth-first expansion to depth 7**, one nesting level per ring size. For a start node `n0`:

```
level 1: all neighbours of n0
level 2: neighbours of those, excluding n0
level 3: neighbours of those, excluding the level-1 parent
         -> if n0 appears here, a 4-branch ring is closed
level 4: ... -> 5-B rings
level 5: ... -> 6-B rings
level 6: ... -> 7-B rings
```

Each level explicitly excludes the immediate parent (preventing backtracking) but **not** earlier ancestors — hence the post-hoc filter that discards any candidate whose node list contains a repeat.

**Deduplication — and a serious defect:**

```
key = product of the ring's node linear indices (excluding n0)
key = key / 10^floor(log10(mean(key)))
unique( round(key, 4) )
```

The product of node indices is a permutation-invariant hash, so a cycle and its reverse collapse together. Elegant in intent. But for the demo volume, node indices run to ~4.6 × 10⁷, so a 6-node product reaches ~10⁴⁶ — vastly beyond IEEE double's exact-integer range of 2⁵³ ≈ 9 × 10¹⁵. After the scaling and `round(·, 4)`, distinct rings routinely collide.

> **Consequence:** ring counts are systematically **under-reported**, non-deterministically, and the error grows with volume size. This is the most serious numerical defect in the toolbox.
>
> **Fix:** canonicalize each ring as a sorted tuple of node IDs and deduplicate on that tuple (a set/dict), or hash the sorted tuple with a proper 64-bit hash. Exact, O(1) per ring, no scaling.

**Also absent (present only as commented-out code):** the flatness test that would reject non-planar rings, and the redundancy filter that would discard higher-B rings composed of lower-B rings. Both are described in Paper II. **You must add them.**

**Complexity:** the unrolled search is O(k⁶) per start node where k is the mean degree (~3.5) — roughly 1,800 paths per node, times ~10³–10⁴ nodes. Tolerable, but there is no memoization and every ring is rediscovered once per participating node, requiring a global dedup pass afterwards.

**Refactor:** `networkx.minimum_cycle_basis(G)` returns a minimal set of independent cycles — which is precisely the paper's "redundancy filter" outcome, obtained for free. For all short cycles up to length 7, use a bounded cycle enumeration with sorted-tuple dedup. Either replaces 500 lines with about five.

#### Ring geometry

**Plane fitting.** `main_ring.m` collects **all branch voxels** of the ring (not just the node positions) and calls `planeFitting_v2`:

```
X = ring voxel coordinates
p = centroid(X)
n = smallest right-singular vector of (X - p)        # total least squares
d = -dot(p, n)
errors = -X*n - d                                    # signed out-of-plane distances
```

This is **total least squares (PCA)** — orientation-independent and correct for vertical planes.

> The repo also contains `planeFitting.m`, an ordinary least-squares fit of `z` on `(x,y)` — exactly the formulation written in Paper II Eqs. (2)–(3), which diverges as the plane approaches vertical. The presence of a `_v2` marked *"Important reference!!!"* shows the authors hit this and fixed it in code without updating the paper. **Use `_v2`. Never port `planeFitting.m`.**

**Ring size.** `ringSize_4E/5E/6E/7E.m` compute polygon **area** by fan triangulation from the first vertex, each triangle's area from **Heron's formula** on the three 3D chord lengths:

$$s_{\triangle} = \tfrac{1}{4}\sqrt{(a+b+c)(-a+b+c)(a-b+c)(a+b-c)}$$

Then $d_{\text{ring}} = 2\sqrt{A/\pi}$ (the equal-area circle diameter).

**Caveats:** (i) fan triangulation of a *non-planar* polygon gives a triangulation-dependent area — project onto the fitted plane first, or use Newell's method; (ii) Heron's formula is numerically unstable for slivers — use the **numerically stable ordered form** or, better, half the norm of the cross product; (iii) `vecdist` is called **without** the `datas` argument in these functions, so they require coordinate triples rather than linear indices — an easily-missed calling-convention switch; (iv) four near-identical hardcoded functions should be one loop over an arbitrary vertex count.

### 3.17 Chain detection

**Code path:** `chain.m`, driver `main_chain.m`.

> ### ⚠ The CAF formula from Paper I is not implemented
>
> Paper I defines a chain alignment factor as a product of dot products weighted by inter-branch angles. **No such computation exists in this repository.** `chain.m` uses a two-predicate threshold rule instead:

```
chain(n, connect, acc, branOrien, datas):
    next_nodes = neighbours of n
    keep those with  coord_dim1(next) > coord_dim1(n)      # monotone along L
                AND  branOrien(branch) <= 60               # orientation gate
    if any survive:
        for each survivor: acc.append(n); recurse into survivor
    else:
        acc.append(n); acc.append(0)                       # 0 = chain terminator
```

So a "chain" is any path that advances monotonically along dim 1 while every branch stays within the orientation gate. Chains are emitted into a flat array delimited by zero sentinels.

**Three defects:**

1. **`connected_node` is not in the repository.** `chain.m` cannot run as shipped. You must write it: return all nodes joined to `n` by a branch (i.e. gather from both columns of `connect`). See §5.
2. **`branOrien(branId)` indexes the wrong column.** `branOrien` is `M × 2` = `[ω, θ]` (§2.4). Linear indexing with a single subscript reads **column 1 = ω**, the 0°–360° azimuth — not θ, the misorientation from L that the 60° gate obviously intends. As written, the gate admits any branch whose azimuth happens to be under 60° regardless of its actual tilt. **Fix: `branOrien(branId, 2)`.**
3. **Monotonicity along dim 1 is a hard directional assumption**, hardcoded. Fine for a specimen aligned to the array axes; wrong for anything else. Parameterize the axis.

**Refactor:** implement the greedy alignment growth described in §12.2 of the companion document — from each seed branch, repeatedly extend to whichever neighbouring branch maximizes `|c_m · c_{m+1}|`, stopping below a cosine threshold. That is direction-agnostic, needs no monotonicity hack, and matches the papers' intent.

### 3.18 3D-FFT of the node array

**Code path:** `main_fft.m`.

```
newskel = zeros(volume_size)
newskel(node_linear_indices) = 1                  # rasterize nodes as a sparse indicator
F = abs( ifftshift( fftn( fftshift( newskel ) ) ) )
F = min-max normalize
display voxels where F > 0.5
```

Confirms the papers: the transform is taken over a **binary node-indicator volume** at the original voxel grid, not over the segmented structure.

**Notes and issues:**
- The shift order is unconventional — the usual convention is `fftshift(fftn(ifftshift(x)))`. For a magnitude spectrum the difference is a phase ramp and is invisible, but fix it for clarity.
- The script contains a **syntax error**: `elsess` instead of `else` (§5). It will not parse.
- The final line calls `open main_dataAnalysis` — wrong filename (the file is `02_main_dataAnalysis.m`).
- No windowing is applied. A rectangular volume boundary produces sinc sidelobes along each axis that can be mistaken for structural peaks. **Apply a Hann/Tukey window** before transforming, or at minimum verify that peaks survive windowing.
- Peak-to-spacing conversion: a peak at reciprocal distance $q$ (in voxel⁻¹) corresponds to a real-space spacing of $1/(q \cdot \text{pixelSize})$ µm. The published values ($d_1 = 0.034\ \mu\text{m}^{-1} \rightarrow 30\ \mu$m) follow this.

---

## 4. Parameter registry

Every tunable in the pipeline, with its location and how to set it. Anything marked **hardcoded** must be lifted into a config file during refactoring.

| # | Parameter | Value in repo | Where | Meaning / how to set |
|---|---|---|---|---|
| 1 | `pixelSize` | 0.65 µm | `02_main_dataAnalysis.m` | voxel size; from the scan |
| 2 | `startPage`/`endPage`/`sigDigits` | 0 / 307 / 4 | `01_main…` | TIFF stack naming |
| 3 | node threshold | `k >= 4` | `node_identification.m` **hardcoded** | ≥3 true skeleton neighbours; leave alone |
| 4 | trimming convergence | node-count stability | `01_main…` | add an iteration cap |
| 5 | `merge_double` guard | 5 voxels (first call), none (second) | `01_main…` | max short-branch length that may be dissolved |
| 6 | `merge_thres` | 10 voxels — **ignored** | `01_main…` | see #7 |
| 7 | `open` ladder | 12 / 10 / 8 voxels at dim-3 < 1700 / < 2200 / else | `merge_nearby.m` **hardcoded** | **the effective $d_m$**; set ≈ mean branch radius, per region |
| 8 | adjacency window | 500 nodes | `merge_connectednodes.m` **hardcoded** | perf hack; replace with KD-tree |
| 9 | proximity window | 1000 nodes | `merge_nearby.m` **hardcoded** | same |
| 10 | `boundaryLimit` | 30 voxels (19.5 µm) | `main_node.m` | ≈ one mean branch length |
| 11 | path-repair gap | 10 voxels | `branch_order.m` **hardcoded** | trailing-voxel sanity threshold |
| 12 | Canny threshold | 0.6 | `main_thickness.m` | surface extraction sensitivity |
| 13 | rough-thickness edge count | ≥5 (`main_thickness`), ≥1 (`branch_thickness`) | **hardcoded** | two inconsistent variants |
| 14 | thickness search sphere | 20 voxels | `main_thickness.m` **hardcoded** | must exceed the largest expected radius |
| 15 | perpendicularity tolerance | 2° | `main_thickness.m` **hardcoded** | controls sample count per cross-section |
| 16 | outlier guard | `rough + 11` voxels | `main_thickness.m` **hardcoded** | rejects far-surface hits |
| 17 | min path length for profiling | 10 voxels | `main_thickness.m` **hardcoded** | branches below this get no profile |
| 18 | fit rejection | \|z\| ≤ 25 µm, 2 µm < L < 35 µm | `main_branchfitting.m` **hardcoded** | paper says 5–40 µm — reconcile |
| 19 | chain orientation gate | 60° | `chain.m` **hardcoded** | and it reads the wrong column (§3.17) |
| 20 | chain direction | increasing dim 1 | `chain.m` **hardcoded** | parameterize |
| 21 | ring max size | 7 branches | `findloop.m` **structural** | unrolling depth; not a variable |
| 22 | ring dedup rounding | 4 decimals | `findloop.m` **hardcoded** | broken by construction (§3.16) |
| 23 | FFT display threshold | 0.5 | `main_fft.m` | visualization only |
| 24 | ring flatness $R^2$ | — | **not implemented** | Paper II specifies ≥ 0.5 |

**Derivation rules for the ones that matter:**

- **#7 ($d_m$)** — measure the mean strut radius from an EDT of the segmented volume, then set $d_m$ to it. Make it a function of position for graded specimens.
- **#10 (boundary)** — one mean branch length, computed from a first pass.
- **#14 (sphere)** — at least 2× the largest expected radius, else the surface is never found.
- **#18 (fit limits)** — from the branch-length distribution: reject beyond, say, mean ± 3 SD, rather than fixed µm literals.

---

## 5. Defect register

Severity: **S1** = prevents execution or corrupts published results · **S2** = silently wrong values · **S3** = performance / maintainability.

| ID | Sev | Location | Defect | Fix |
|---|---|---|---|---|
| D1 | **S1** | `chain.m` | calls `connected_node()`, **which does not exist in the repository** — chain analysis cannot run | write it: gather neighbours from both columns of `connect` |
| D2 | **S1** | `branch_thick.m` | calls `chooseregion()`, **not in the repository** | unused by the main path; delete the file or implement a radius query |
| D3 | **S1** | `main_fft.m` line 31 | `elsess` — **syntax error**, file will not parse | `else` |
| D4 | **S1** | `findloop.m` | ring dedup hashes by a **product of node indices** that overflows double precision (~10⁴⁶ vs 2⁵³); distinct rings collide and are discarded | deduplicate on sorted node-ID tuples |
| D5 | **S1** | `main_branchfitting.m` | calls `vecdist(a,b)` with 2 arguments; the function needs 3 (`datas`) → runtime error. Also hardcodes `0.65` instead of `pixelSize` | use `General/main_profile.m`, which fixes both |
| D6 | **S2** | `main_interBranchAngle.m` | measures only **cyclically adjacent** neighbour pairs → 4 angles at an N-4 node instead of 6, in arbitrary order | enumerate all $\binom{n}{2}$ pairs |
| D7 | **S2** | `chain.m` | `branOrien(branId)` linear-indexes an M×2 array → reads **ω** (azimuth) where **θ** (misorientation) is intended | `branOrien(branId, 2)` |
| D8 | **S2** | `myangle.m` | returns `phi = 0` when `Δd1 == 0` and `theta = 0` when `Δd3 == 0`; branches exactly perpendicular to L get θ = 0° instead of 90° | use `atan2`; handle the degenerate cases correctly |
| D9 | **S2** | `merge_connectednodes.m` | places the merged node at the **maximum linear index**, not the cluster centroid as the paper states | use the centroid, as `merge_nearby.m` does |
| D10 | **S2** | `merge_connectednodes.m`, `merge_nearby.m` | fixed 500/1000-node lookahead windows assume spatial locality in linear-index order; neighbours separated along dim 3 differ in index by `nx·ny` (148,996 for the demo volume) and **are missed** | KD-tree + union–find over the proximity graph |
| D11 | **S2** | `main_interBranchAngle.m` | `real()`/`abs()` applied to complex `acos` output silently maps out-of-range results to 0° or 180° | clamp the argument to [−1, 1] |
| D12 | **S2** | `offangle.m` | crosses two unit vectors and treats the result as unit-length; the `acos` argument is not a cosine | do not port; use `ori3.m` |
| D13 | **S2** | `main_thickness.m` | no bounds check on `cont(x0-w : x0+w, …)` → index error near volume faces | pad the volume, or clamp the window |
| D14 | **S2** | `main_thickness.m` | branch axis approximated by the **endpoint chord**, so the "perpendicular" plane is misaligned for curved branches | use a local tangent |
| D15 | **S2** | `main_thickness.m` | histogram bins hardcoded to `300:50:500` and `linspace(300,500,12)` — stale values incompatible with µm-scale thickness | derive bins from the data |
| D16 | **S2** | `main_thickness.m` | references undefined `color` (colormap is `color1`) | rename |
| D17 | **S2** | `merge_nearby.m` | the `open` flag **silently overrides** the caller's `merge_thres` | remove the flag; pass `d_m` explicitly |
| D18 | **S3** | `main_thickness.m` | allocates a full-volume temporary **per centerline voxel** — the dominant cost of the pipeline | crop a local window, or use an EDT / KD-tree |
| D19 | **S3** | `neighbor.m` | rebuilds a `meshgrid` and loops `sub2ind` for every voxel query | vectorize; precompute offset table |
| D20 | **S3** | `01_main…`, most `main_*.m` | arrays grown by concatenation inside loops (`x = [x; new]`) | preallocate |
| D21 | **S3** | `branch_sort2.m` | writes to `connect` before it exists; errors if early segments all fail | preallocate / initialize empty |
| D22 | **S3** | `main_*.m` | computation and plotting interleaved; `saveas` calls fire mid-analysis; scripts end by `open`-ing the next script in the editor | separate compute from render; return values, don't plot |
| D23 | **S3** | `myanglez.m`, `planeFitting.m`, `survalue.m`, `branchz.m` | rely on `global datas` while the rest of the toolbox passes it explicitly | make all signatures explicit |
| D24 | **S3** | two `findloop.m`, `main_profile.m`/`main_branchfitting.m`, `branch_thick.m`/`branch_thickness.m`, four `ringSize_*E.m` | duplicated/divergent implementations of the same function | keep one of each |

**Minimum patch set to make the shipped demo run end to end:** D1, D3, D5. **Minimum set for trustworthy numbers:** add D4, D6, D7, D8, D17.

---

## 6. Numerical and convention hazards

1. **`interangle` returns radians**; every caller must multiply by `180/π`. Nothing in the signature says so.
2. **`vecdist` has two calling conventions** — `(idx1, idx2, datas)` for linear indices, `(xyz1, xyz2)` for coordinate triples. The `ringSize_*` functions use the second; everything else uses the first. Silently wrong if mixed.
3. **`node(:,2)` changes meaning mid-pipeline** (§2.2).
4. **`branOrien` column order is `[ω, θ]`** — the reverse of how the papers list them (§2.4).
5. **All thicknesses are radii, not diameters** (§3.11).
6. **All lengths are in voxels until a driver multiplies by `pixelSize`.** Mixed-unit intermediate variables are everywhere; several magic constants (10, 11, 20, 25, 35) are voxels in some places and µm in others.
7. **`acos` domain overflow** produces complex results (D11).
8. **Node identity is a voxel index**, so merging changes identity. Any external reference to a node captured before merging is invalid afterwards.
9. **Isotropic voxels assumed throughout.** `vecdist` computes an unweighted Euclidean norm on index differences. Anisotropic voxels require scaling every distance.
10. **Order dependence.** Merging results depend on the ascending-linear-index iteration order. Two runs on the same data agree, but the same structure imaged in a different orientation may not merge identically.

---

## 7. Performance profile and remedies

Measured against the demo volume (386 × 386 × 308 ≈ 4.6 × 10⁷ voxels).

| Stage | Complexity as written | Bottleneck | Remedy | Expected gain |
|---|---|---|---|---|
| I/O | O(N), unpreallocated | array growth, `double` storage | preallocate, use `bool` | 8–64× memory |
| Skeletonization | O(iter × N_solid) | already efficient | swap in `skimage` | parity, less code |
| Node ID | O(N_skel × 27), scalar loop | `neighbor.m` | 3×3×3 convolution | ~10³× |
| Trimming | O(passes × N × 27) + `bwlabeln` per pass | full-volume relabel each pass | do it on the graph | ~10²× |
| Registration | O(N_skel × 27) | as above | `skan`-style single pass | ~10× |
| `merge_connectednodes` | O(N_node × 500) | windowed scan | KD-tree + union–find | ~10× and *correct* |
| `merge_nearby` | O(N_node × 1000) | windowed scan | KD-tree + union–find | ~10× and *correct* |
| **Thickness** | **O(N_sample × N_volume)** | **full-volume temporary per sample point** | **EDT, once** | **~10⁴–10⁶×** |
| Rings | O(N_node × k⁶) | no memoization; rediscovery | bounded cycle enumeration + set dedup | ~10× and *correct* |
| FFT | O(N log N) | fine | add windowing | — |

**The single highest-value change is the thickness stage (D18).** It dominates total runtime by orders of magnitude and is replaceable by one distance-transform call.

**Scaling to Paper II volumes.** A 0.32 mm³ volume at 1.3 µm is ~1.5 × 10⁸ voxels — over 3× the demo. The window-based merges (D10) degrade in *correctness*, not just speed, as node counts grow. Chunk the volume into overlapping sub-volumes (overlap ≥ 2 × mean branch length), build a graph per chunk, and stitch by matching nodes in the overlap region. This is also how you get the per-sub-volume statistics Paper II reports.

---

## 8. Porting map

Dependency order — each row depends only on rows above it. Signatures are the recommended ported API, not the MATLAB ones.

| Order | Module | Replaces | Suggested implementation |
|---|---|---|---|
| 1 | `io.load_volume(path, voxel_size, axes) -> Volume` | `01_main…` §1 | `tifffile`, `dask` for large stacks |
| 2 | `skeleton.skeletonize(Volume) -> ndarray[bool]` | `Skeleton3D.m` + 5 helpers (**875 lines**) | `skimage.morphology.skeletonize(method='lee')` |
| 3 | `graph.build(skel) -> (Nodes, Branches)` | `node_identification`, `branch_trim`, `branch_sort2`, `branch_order` | `skan.Skeleton` gives ordered paths + connectivity directly |
| 4 | `graph.trim_dangling(G) -> G` | trimming loop | iteratively drop degree-1 nodes |
| 5 | `graph.dissolve_degree2(G, max_len=None) -> G` | `merge_double.m` | splice incident edges; keep the length guard |
| 6 | `graph.merge_clusters(G, d_m) -> G` | `merge_connectednodes`, `merge_nearby` | `scipy.spatial.cKDTree.query_pairs` → union–find → centroid |
| 7 | `graph.drop_parallel_edges(G) -> G` | `DeleteRepeatedBran.m` | keep min-tortuosity edge per node pair |
| 8 | `geom.lengths(G) -> length, distance, tortuosity` | `branch_length.m` | cumulative path distance + node stubs |
| 9 | `geom.thickness(G, Volume) -> profiles` | `main_thickness.m` §1–2 (**the hot spot**) | `scipy.ndimage.distance_transform_edt`, sample along paths |
| 10 | `geom.fit_profile(profile, model) -> params` | `main_branchfitting`/`main_profile` | `numpy.polyfit` (quadratic) and `scipy.optimize.curve_fit` (Paper II quartic) |
| 11 | `geom.orientation(vec, axes) -> theta, omega` | `myangle`, `vectorAngle` | `atan2`; return the **paper's** naming |
| 12 | `nodes.interbranch_angles(G) -> dict` | `interangle`, `main_interBranchAngle` | all pairs; clamp `acos` |
| 13 | `nodes.planarity(G) -> p, normal` | `ori3.m` | plane through neighbours **or** equiangular axis — pick and document |
| 14 | `rings.find(G, max_size=7) -> Rings` | `findloop.m` (**309 lines**) | `networkx.minimum_cycle_basis`, or bounded enumeration + tuple dedup |
| 15 | `rings.geometry(Rings, G) -> normal, area, d_ring, flatness` | `planeFitting_v2`, `ringSize_*E` | SVD plane fit; project; shoelace or Newell area |
| 16 | `chains.find(G, axis, cos_thresh) -> Chains` | `chain.m` (**broken**) | greedy alignment growth (§3.17) |
| 17 | `spectral.node_fft(G, Volume) -> spectrum` | `main_fft.m` | rasterize, window, `numpy.fft.fftn` |
| 18 | `export.beam_model(G) -> mesh/deck` | **not in repo** | write from Paper II §2.4 |
| 19 | `viz.*` | `plot/`, plotting halves of `main_*.m` | `pyvista` / `matplotlib`; keep separate from compute |

**Third-party equivalents worth adopting wholesale:** `skimage` (step 2), `skan` (step 3), `scipy.spatial` (step 6), `scipy.ndimage` (step 9), `networkx` (step 14). Together these subsume roughly **half the repository**.

---

## 9. Test plan

### 9.1 Synthetic phantoms with known answers

Build these as binary volumes and run the full pipeline. Each has an analytic expectation.

| Phantom | Expected result | Catches |
|---|---|---|
| **Single straight cylinder**, radius r, length L | 1 branch, 0 nodes (or 2 endpoints); tortuosity = 1.000; thickness = r everywhere | length accounting, radius-vs-diameter, unit conversion |
| **Single bent cylinder** (circular arc, known chord) | tortuosity = arc/chord, exact | path ordering, curved-length accumulation |
| **Hourglass strut** (analytic quartic waist) | fitted `a` recovers the input parameter | profile extraction + fitting |
| **Simple cubic lattice** | all nodes N-6; all inter-branch angles 90° or 180°; all rings 4-B; ring size = pitch | node typing, angle enumeration, ring detection |
| **Diamond (tetrahedral) lattice** | all nodes N-4; all angles 109.47°; rings 6-B | N-4 angle enumeration (**catches D6**) |
| **2D honeycomb extruded** | N-3 nodes; angles 120°; planarity ≈ 0; rings 6-B | planarity index, N-3 statistics |
| **Kelvin (tetrakaidecahedral) foam** | mean connectivity 4; rings 4-B and 6-B in 3:1 count ratio | ring dedup (**catches D4**) |
| **Lattice rotated 45° off-axis** | identical statistics to the unrotated case | axis-convention leakage (**catches D8, D20**) |
| **Lattice with one strut deleted** | exactly one N-3 pair appears; ring count drops predictably | defect sensitivity |
| **Lattice scaled 2×** | all lengths and thicknesses exactly double; angles, tortuosity, node ratios unchanged | unit handling, hardcoded voxel constants (**catches #11, #14, #16, #18**) |

The **rotation** and **scaling** phantoms are the highest-yield tests: together they expose nearly every hardcoded voxel literal and every axis assumption in the codebase.

### 9.2 Invariants to assert at every stage

```
assert  sum(i * count_of_N_i_nodes) == 2 * n_branches    # handshake, modulo boundary
assert  all(tortuosity >= 1.0)
assert  all(t_min <= t_cen <= t_mean)
assert  all(0 <= theta <= 90)  and  all(0 <= omega < 360)
assert  all(mean_interbranch_angle(N3) <= 120.0)          # geometric necessity
assert  n_angles(node) == degree * (degree - 1) / 2       # catches D6
assert  no self-loops:      branch.node_a != branch.node_b
assert  no parallel edges after dedup
assert  solid phase is ONE connected component (bicontinuity)
assert  every branch path is contiguous (max step <= sqrt(3) voxels)  # catches D8/§3.8 jumps
assert  planarity index in [0, 1]
```

### 9.3 Regression targets

Run on the shipped demo volume and compare against Paper I's published values (companion document §10.1). Expect agreement to within a few percent on:

- node type fractions ≈ 50 / 35 / 12 / 3 %, ratio ≈ 17 : 12 : 4 : 1
- N-3 mean inter-branch angle ≈ 118.1° ± 3.5°
- N-4 mean ≈ 110.0° ± 9.4° — **expect a discrepancy here** until D6 is fixed
- mean physical length 21.7 ± 7.9 µm; Euclidean 18.0 ± 6.2 µm; ratio 1.2 ± 0.1
- mean thickness 6.2 ± 1.5 µm (**as a radius**)
- ring counts 213 / 577 / 614 / 476 — **expect under-counting** until D4 is fixed
- mean ring diameter 30.0 ± 9.6 µm; mean branches per ring ≈ 5.7

Deviations on the N-4 angles and the ring counts are *expected* and are in fact the signature that D6 and D4 were real.

---

## 10. Refactoring brief

*This section is written to be handed to a coding agent verbatim.*

### 10.1 Objective

Port the MATLAB toolbox to a tested, documented Python package that (a) reproduces Paper I's published statistics on the demo volume, (b) fixes the defects in §5, (c) adds the Paper II descriptors that are missing, and (d) scales to volumes of ~10⁸ voxels.

### 10.2 Non-negotiable requirements

1. **Separate computation from visualization.** No plotting inside any analysis function. Analysis functions return data structures.
2. **No global state.** Every function takes its inputs explicitly (removes `global datas`).
3. **Explicit units.** Carry `voxel_size` in the `Volume` object; convert to µm once, at the boundary. No bare voxel literals in analysis code.
4. **Explicit axis mapping.** The `Volume` object declares which array dimension is L, C, R. All orientation code reads it. No hardcoded dim-1/dim-3 assumptions.
5. **Stable integer node IDs**, decoupled from voxel positions (§2.5).
6. **All parameters in one config object**, defaulting to the §4 table, with derivation helpers (`d_m_from_radius()`, `boundary_from_length()`).
7. **Angle naming follows the papers** — θ = misorientation from L, ω = in-plane azimuth. Not the code's swapped convention.
8. **Thickness reported as radius and diameter, explicitly named.** Never a bare `thickness`.

### 10.3 Work order

**Phase 1 — scaffold and validate.** Steps 1–3 of §8. Build the phantom suite (§9.1) and the invariant assertions (§9.2) *first*. Confirm the graph builder passes the cubic and diamond lattice tests before proceeding.

**Phase 2 — cleaning.** Steps 4–7. Replace windowed scans with KD-tree + union–find (fixes D10). Use centroid placement everywhere (fixes D9). Lift `d_m` to config with position-dependent support (fixes D17). Re-run the phantoms.

**Phase 3 — descriptors.** Steps 8–13. Replace the thickness stage with a distance transform (fixes D18, D13, D14). Enumerate all angle pairs (fixes D6). Use `atan2` (fixes D8). Clamp `acos` (fixes D11). Validate against the §9.3 regression targets.

**Phase 4 — rings and chains.** Steps 14–16. Tuple-based dedup (fixes D4). Add the flatness filter and the redundancy filter from Paper II. Implement greedy alignment chains (fixes D1, D7).

**Phase 5 — Paper II extensions.** The quartic profile model; per-sub-volume batching with overlap stitching; relative-density computation; the beam-model exporter (§8 step 18).

### 10.4 Acceptance criteria

- All phantoms in §9.1 pass within tolerance.
- All invariants in §9.2 hold on the demo volume.
- Regression targets in §9.3 match, with documented and justified deviations for N-4 angles and ring counts.
- The full demo volume processes in minutes, not hours.
- The 2× scaling phantom returns exactly doubled lengths — proving no hardcoded voxel constants remain.
- The 45°-rotation phantom returns statistics identical to the unrotated case — proving no axis assumptions remain.

### 10.5 Explicit "do not port" list

`offangle.m` (mathematically incorrect), `planeFitting.m` (superseded by `_v2`; use TLS), `branch_thick.m` (broken dependency, superseded), `network_tools/findloop.m` (older duplicate), `main_branchfitting.m` (use `main_profile.m`), `myanglez.m`/`branchz.m`/`survalue.m` (global-state legacy), and the entire `toolbox/skeleton/` directory (use `skimage`).

---

## 11. Quick-reference: pipeline call graph as shipped

```
01_main_dataProcessing.m
├── load TIFF stack                          -> orig  [nx,ny,nz]
├── Skeleton3D(orig)                         -> skel_c
│     └── FillEulerLUT, p_EulerInv, p_is_simple, p_oct_label, pk_get_nh*
├── LOOP until node count stabilizes:
│     ├── node_identification(skel)          -> node [N x 2]  (voxel_degree)
│     └── branch_trim(node, skel)            -> trimskel
├── branch_sort2(trimskel, trimnode, datas)  -> bran{}, connect[M x 2], node (TRUE degree)
├── merge_connectednodes(...)                -> c2, n2, b2
├── merge_double(..., 5)                     -> c3, n3, b3     [N-2 dissolution, guarded]
├── merge_nearby(..., 10, 1, datas)          -> c4, n4, b4     [d_m clustering; `1` overrides `10`]
├── merge_double(...)                        -> c5, n5, b5     [N-2 dissolution, unguarded]
└── save orig, trimskel, node, bran, connect

02_main_dataAnalysis.m   (dispatcher; each script `open`s the next)
├── main_node.m           -> isboundary(30) ; node-type histogram
├── main_length.m         -> isboundary ; branch_order ; branch_length ;
│                            DeleteRepeatedBran ; branch_length again
├── main_thickness.m      -> edge3 ; rough cube search ; sphere+2deg profile ; t_min
│     └── main_branchfitting.m / main_profile.m  -> quadratic fit, pooled normalization
├── main_branchOrientation.m -> myangle -> branOrien [omega, theta]
├── main_interBranchAngle.m  -> interangle (cyclic pairs only)
├── main_correlation.m       -> cross-descriptor scatter/polar plots
├── main_chain.m             -> chain.m  [BROKEN: connected_node missing]
├── main_ring.m              -> findloop ; planeFitting_v2 ; ringSize_{4,5,6,7}E
│     └── main_ring_plot.m
└── main_fft.m               -> node rasterization + fftn  [BROKEN: syntax error]
```

Legend: **bold-path stages** carry the S1/S2 defects catalogued in §5.

# Custom HPC Mixed-FEM Solver Design and Implementation Plan

Status: DESIGN BASELINE  
Scope: Q2/Q1 patient-specific brain mechanics and reusable future PDE problems  
FEM execution: Not started by this document

## 1. Design objective

Develop a production-quality C++ solver framework with direct control over MPI ownership, halo exchange, CPU/GPU execution, memory, partitioning, Krylov methods, and block preconditioners, while retaining MFEM/HYPRE as an independent numerical reference.

The framework must support the current mixed formulation without changing its mathematics:

\[
\begin{bmatrix} A & B^T \\ B & -C \end{bmatrix}
\begin{bmatrix} u \\ p \end{bmatrix}
=
\begin{bmatrix} f \\ g \end{bmatrix},
\]

with

\[
A(u,v)=\int 2\mu\,\epsilon(u):\epsilon(v)\,dx,
\qquad C=(1/\lambda)M_p.
\]

The \(\lambda\operatorname{div}(u)\operatorname{div}(v)\) term must not be inserted into the mixed-system displacement block \(A\).

## 2. Guiding principles

1. The full-resolution voxel-conforming patient mesh remains authoritative.
2. The Q2/Q1 mixed pair and material model remain unchanged.
3. MFEM/HYPRE remains the reference backend.
4. Every custom kernel is validated against the reference backend before production use.
5. Ownership, communication, memory, and accelerator placement are explicit framework concepts.
6. Resource ownership uses RAII and deterministic lifetime management.
7. MPI rank count, thread count, GPU mapping, and partitioning are configuration data, not hidden defaults.
8. Solver results require independently recomputed global residuals.
9. Validation meshes and reduced models are not substitutes for the authoritative full-resolution model.
10. FEM production execution begins only after meshing/certification gates are satisfied.

## 3. Two-backend architecture

### 3.1 Reference backend

Use MFEM for finite-element spaces, coefficient evaluation, partial assembly, true-DOF mappings, and operator actions. Use HYPRE for distributed sparse auxiliary operators and BoomerAMG.

This backend establishes expected values for:

- A, B, \(B^T\), and C actions;
- boundary elimination and reconstruction;
- Q2-to-Q1 transfer;
- Schur action and inverse application;
- block-preconditioner action;
- residual histories and final solutions.

### 3.2 Custom backend

Implement a separate C++ backend behind common interfaces. The custom backend may progressively replace:

- mesh/DOF storage;
- Q2 element kernels;
- coefficient and quadrature kernels;
- vector operations;
- halo exchange;
- transfer operators;
- Krylov solvers;
- pressure-mass preconditioner;
- auxiliary elasticity preconditioner;
- multilevel coarse operators.

The custom backend must be able to run beside the reference backend on the same mesh, partition, right-hand side, and initial vector.

## 4. Software layers

### Layer 0: configuration and provenance

Components:

- YAML/JSON configuration reader;
- immutable run configuration object;
- source and mesh SHA256 verification;
- material and boundary-condition manifest;
- run ID and checkpoint metadata;
- software/version capture.

### Layer 1: resource ownership

RAII classes:

- `MpiSession`;
- `MpiCommunicator`;
- `ThreadRuntime`;
- `CudaContext`;
- `DeviceBuffer<T>`;
- `HostBuffer<T>`;
- `PinnedBuffer<T>`;
- `MeshHandle`;
- `DofMapHandle`;
- `OperatorHandle`;
- `PreconditionerHandle`;
- `CheckpointHandle`.

No raw ownership of MPI requests, CUDA streams, device allocations, or file handles should escape these classes.

### Layer 2: mesh and partition

Responsibilities:

- read authoritative HEX8/QUAD4 mesh;
- preserve material and interface attributes;
- construct global element/node identifiers;
- build element adjacency and face ownership;
- construct partition-local cells and halo entities;
- verify no duplicate or non-manifold volume faces;
- maintain global-to-local and local-to-global maps.

### Layer 3: partitioning

Supported strategies:

1. geometry-aware baseline partition;
2. serial METIS partition;
3. distributed ParMETIS partition;
4. optional PT-Scotch backend.

For each partition record:

- elements per rank;
- vertices/DOFs per rank;
- material workload per rank;
- interface/halo entities;
- edge cut and communication volume;
- maximum/mean imbalance;
- partition construction time.

For multi-node production, ParMETIS is the preferred distributed path when available. METIS remains useful for single-process partition construction and debugging.

### Layer 4: finite-element spaces and DOF maps

Implement or wrap:

- H1 Q2 vector displacement space;
- continuous H1 Q1 scalar pressure space;
- Q1 vector auxiliary space;
- true/free/essential DOF maps;
- boundary-DOF canonicalization;
- global numbering and ownership;
- Q2-to-Q1 and Q1-to-Q2 transfer maps.

Required invariants:

\[
N_{u,free}+N_{u,essential}=N_{u,total},
\]

and transfer dimensions must match the actual local and global spaces.

### Layer 5: operator kernels

Implement common `LinearOperator` and `BlockOperator` interfaces with:

- `apply(x, y)`;
- `apply_transpose(x, y)` where required;
- `energy(x)`;
- `diagonal()` where available;
- `communicator()`;
- `global_size()` and `local_size()`;
- `backend()`;
- `memory_bytes()`.

Operators:

- matrix-free heterogeneous Q2 A;
- divergence B;
- transpose \(B^T\);
- pressure mass C;
- auxiliary Q1 elasticity operator;
- scaled pressure mass Schur approximation;
- complete mixed block operator.

The Q2 A kernel must use heterogeneous \(\mu(x)\) and the symmetric-strain form only.

### Layer 6: communication

Use explicit communication plans:

- neighbor rank lists;
- send/receive index maps;
- persistent MPI requests where beneficial;
- nonblocking halo exchange;
- overlap of communication and element kernels;
- MPI datatype or packed-buffer policy;
- GPU-aware MPI capability detection.

Communication diagnostics must measure:

- bytes sent/received;
- number of messages;
- halo exchange time;
- wait time;
- overlap ratio;
- rank-to-rank imbalance.

### Layer 7: memory and execution backends

CPU backend:

- OpenMP-parallel element loops;
- NUMA-aware first-touch initialization;
- aligned and contiguous storage;
- optional thread pinning;
- reduction policies with deterministic debug mode.

CUDA backend:

- device-resident vectors and coefficients;
- CUDA streams;
- fused element action kernels;
- asynchronous halo transfers where supported;
- explicit host/device ownership state;
- GPU-aware MPI detection and fallback.

The first custom implementation should be CPU MPI/OpenMP. CUDA is added only after CPU/reference equivalence is established.

## 5. Solver algorithms

### 5.1 Inner A solve

Use PCG only when the preconditioner is demonstrably SPD-compatible. The custom inner solver must record:

- initial residual;
- residual at iterations 10, 25, 50;
- final residual;
- iteration count;
- monotonicity;
- energy positivity;
- breakdown status.

The MFEM/HYPRE A-inner result is the reference.

### 5.2 Schur approximation

Use the fixed model:

\[
\widehat S = \left(1/\lambda+c/\mu\right)M_p,
\qquad c=0.1947265359680029.
\]

No retuning is permitted in the reference comparison. Implement the pressure mass action and scalable inverse without dense storage.

### 5.3 Outer solve

Use FGMRES with the block-triangular preconditioner because the inner A solve is variable/inexact. Record both reported and independently recomputed residuals.

### 5.4 Custom Krylov implementation

Implement:

- distributed dot products;
- norm reductions;
- restart policy;
- flexible basis storage;
- finite-value checks;
- convergence and stagnation tests;
- bounded iteration limits;
- checkpoint/restart of Krylov state where practical.

Compare custom FGMRES against MFEM/HYPRE on identical systems.

## 6. Preconditioner roadmap

### Stage A: reference preconditioner

- Q1 heterogeneous auxiliary sparse operator;
- HYPRE BoomerAMG elasticity configuration;
- measured setup/application memory;
- no parameter sweep during production.

### Stage B: custom wrapper

Keep HYPRE AMG but own the block application, transfer, vectors, logging, and memory policy.

### Stage C: custom transfer and mass inverse

Replace Q2/Q1 transfer and pressure-mass inverse while retaining HYPRE auxiliary AMG.

### Stage D: custom multilevel auxiliary preconditioner

Only after reference agreement, investigate a custom geometric/algebraic multilevel method. It must report operator/grid complexity, coarse sizes, setup memory, and PCG compatibility.

### Stage E: accelerator path

Port the verified operator and transfer kernels to CUDA. Retain CPU fallback and reference comparisons.

## 7. METIS/ParMETIS integration plan

### Inputs

- HEX8 element adjacency;
- shared-node graph;
- optional material/interface weights;
- target MPI rank count.

### Outputs

- global element-to-rank map;
- local element lists;
- ghost/halo map;
- boundary/interface ownership;
- partition metadata.

### Validation

Compare geometry-aware, METIS, and ParMETIS partitions using:

- maximum element imbalance;
- maximum DOF imbalance;
- halo size;
- edge cut;
- memory imbalance;
- A-application time;
- communication time;
- total FGMRES time.

Partitioning must never change element connectivity, material labels, or global mathematical operators.

## 8. Verification hierarchy

### Level 1: unit tests

- RAII destruction and repeated construction;
- MPI communicator lifecycle;
- index-map bounds;
- vector operations;
- face ownership;
- partition maps;
- Q2/Q1 transfer dimensions;
- material coefficient lookup;
- boundary DOF canonicalization.

### Level 2: manufactured problems

- scalar and vector polynomial fields;
- exact element integration checks;
- zero operator action;
- symmetry checks;
- positive energy checks;
- known divergence and pressure-mass actions.

### Level 3: reference equivalence

For small and medium meshes compare custom and MFEM/HYPRE:

- A, B, \(B^T\), C outputs;
- transfer outputs;
- block residuals;
- preconditioner outputs;
- FGMRES solution vectors;
- global residual norms.

Targets must be stated by operation and precision. Debug mode should use deterministic reductions.

### Level 4: validation patient mesh

Use the certified validation-scale geometry before full-resolution FEM. Confirm all five materials, support BCs, load, transfers, block actions, and reconstruction.

### Level 5: full-resolution readiness

Do not declare production solver readiness until:

- reference solution is finite;
- custom/reference residuals agree within documented tolerance;
- constrained DOFs reconstruct correctly;
- pressure and displacement fields are finite;
- memory peaks are measured;
- restart and failure behavior are tested;
- multi-rank runs pass on the intended cluster.

## 9. Performance instrumentation

Record per stage:

- wall time;
- CPU time;
- peak RSS and device memory;
- MPI bytes/messages;
- OpenMP thread count;
- GPU kernel and transfer time;
- operator applications;
- preconditioner applications;
- reductions;
- solver iterations;
- residual history.

Use a machine-readable JSON/CSV event schema. Do not mix decimal GB and GiB.

## 10. Memory model

Track separately:

- replicated metadata;
- distributed mesh and DOF maps;
- Q2 coefficient/quadrature data;
- pressure data;
- auxiliary sparse matrix;
- AMG hierarchy;
- transfer maps;
- Krylov basis;
- work vectors;
- temporary setup allocations;
- device copies.

The report must show aggregate and per-rank memory. Increasing ranks on one node does not add physical RAM.

## 11. Checkpoint and restart design

Checkpoint:

- run configuration and hashes;
- mesh/partition identifiers;
- completed setup stages;
- solver iteration and residual history;
- solution vectors where restart is supported;
- failure reason and rank;
- memory/timing summary.

Writes must use temporary files, fsync, and atomic rename. Restart must reject mismatched mesh, partition, material, or solver configuration hashes.

## 12. Repository layout

Suggested implementation layout:

```text
solver/
  include/brain_solver/
    core/
    mpi/
    mesh/
    partition/
    dof/
    operators/
    transfer/
    preconditioners/
    krylov/
    io/
  src/
  tests/
  benchmarks/
  configs/
  scripts/
```

The existing MFEM/HYPRE validation programs remain under `solver_validation/` and are not overwritten by the custom implementation.

## 13. Implementation sequence

1. Freeze the certified Full774 mesh and complete the 200-patient meshing checkpoint.
2. Define C++ interfaces, configuration schema, provenance, and RAII resource classes.
3. Implement MPI vector ownership and halo exchange with unit tests.
4. Implement custom distributed vector operations and deterministic reductions.
5. Implement custom Q2 A partial-assembly kernel on a small mesh.
6. Compare A action and energy against MFEM.
7. Implement B, \(B^T\), C, and pressure-mass actions.
8. Implement Q2/Q1 transfer and essential-DOF handling.
9. Implement custom PCG and FGMRES against reference outputs.
10. Integrate METIS, then ParMETIS, with partition diagnostics.
11. Integrate HYPRE as the initial auxiliary AMG backend.
12. Validate the complete custom block solver on the coarse patient geometry.
13. Add OpenMP optimization and NUMA measurements.
14. Add CUDA kernels and GPU-aware communication only after CPU equivalence.
15. Investigate replacement AMG/multilevel methods only after the reference path is stable.
16. Run full-resolution FEM only after explicit solver-readiness approval.

## 14. Acceptance gates

### Mathematical gate

- exact Q2/Q1 formulation preserved;
- no \(\lambda\operatorname{div}u\operatorname{div}v\) term inserted into A;
- fixed Schur constant retained;
- heterogeneous materials preserved.

### Numerical gate

- operator actions agree with reference;
- energy and symmetry checks pass;
- inner PCG is mathematically compatible;
- outer true residual is independently verified.

### Distributed gate

- MPI ownership and halo checks pass;
- no duplicated or missing DOFs;
- partition imbalance is documented;
- restart works after interruption.

### Performance gate

- memory is below the configured safety threshold;
- communication is measured;
- CPU/GPU speedups are measured against the reference;
- no optimization changes numerical results beyond tolerance.

## 15. Final strategy

The project will maintain two trustworthy paths:

1. MFEM/HYPRE reference path for independent correctness and regression checking.
2. Custom MPI/OpenMP/CUDA path for ownership, performance, and architectural control.

The custom path may eventually become the primary production solver, but only after it reproduces the reference path on controlled problems and passes the mathematical, numerical, distributed, and performance gates above.

This design intentionally separates exact medical-geometry preservation from solver optimization. Computational savings must not be obtained by silently changing the authoritative mesh or material labels.

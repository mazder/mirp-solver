#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <set>
#include <vector>
#include <numeric> // for accumulate
#include <chrono>

using namespace std;
using namespace mfem;

void PrintMeshAttributes(const mfem::Mesh &mesh)
{
    using namespace mfem;

    // --- Element (volume) attributes
    std::set<int> elem_attrs;
    int NE = mesh.GetNE();
    for (int i = 0; i < NE; i++)
    {
        elem_attrs.insert(mesh.GetAttribute(i));
    }

    std::cout << "Element (volume) attribute values present in mesh: ";
    for (auto attr : elem_attrs)
        std::cout << attr << " ";
    std::cout << std::endl;

    // --- Boundary attributes
    std::set<int> bdr_attrs;
    int NB = mesh.GetNBE();
    for (int i = 0; i < NB; i++)
    {
        bdr_attrs.insert(mesh.GetBdrAttribute(i));
    }

    std::cout << "Boundary attribute values present in mesh: ";
    for (auto b : bdr_attrs)
        std::cout << b << " ";
    std::cout << std::endl;
}

// Compute strain and stress tensors per element (averaged over integration points)
void ComputeStrainStress(FiniteElementSpace &fespace,
                         const GridFunction &x,
                         double lambda, double mu,
                         std::vector<DenseMatrix> &strain,
                         std::vector<DenseMatrix> &stress)
{
    Mesh *mesh = fespace.GetMesh();
    int dim = mesh->Dimension();
    int NE = mesh->GetNE();

    strain.resize(NE);
    stress.resize(NE);

    for (int i = 0; i < NE; i++)
    {
        const FiniteElement *fe = fespace.GetFE(i);
        const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), fe->GetOrder());

        strain[i].SetSize(dim, dim);
        stress[i].SetSize(dim, dim);
        strain[i] = 0.0;
        stress[i] = 0.0;

        Array<int> vdofs;
        fespace.GetElementVDofs(i, vdofs);
        Vector el_dofs(vdofs.Size());
        x.GetSubVector(vdofs, el_dofs);

        DenseMatrix dshape(dim, fe->GetDof());

        double w_sum = 0.0;
        DenseMatrix eps_sum(dim), sigma_sum(dim);
        eps_sum = 0.0;
        sigma_sum = 0.0;

        for (int j = 0; j < ir.GetNPoints(); j++)
        {
            const IntegrationPoint &ip = ir.IntPoint(j);

            fe->CalcDShape(ip, dshape);

            DenseMatrix grad_u(dim);
            grad_u = 0.0;

            // Calculate displacement gradient grad_u
            for (int k = 0; k < fe->GetDof(); k++)
            {
                for (int d = 0; d < dim; d++)
                {
                    // Note: careful indexing of components
                    int comp = d;
                    double val = el_dofs(k * dim + comp);
                    for (int c = 0; c < dim; c++)
                    {
                        grad_u(d, c) += val * dshape(c, k);
                    }
                }
            }

            // Compute strain tensor ε = 0.5 * (grad_u + grad_u^T)
            DenseMatrix eps_local(dim);
            for (int r = 0; r < dim; r++)
                for (int c = 0; c < dim; c++)
                    eps_local(r, c) = 0.5 * (grad_u(r, c) + grad_u(c, r));

            double trace = 0.0;
            for (int d = 0; d < dim; d++) trace += eps_local(d, d);

            // Compute stress tensor σ = 2μ ε + λ tr(ε) I
            DenseMatrix sigma_local(dim);
            for (int r = 0; r < dim; r++)
                for (int c = 0; c < dim; c++)
                    sigma_local(r, c) = 2.0 * mu * eps_local(r, c) + lambda * trace * (r == c ? 1.0 : 0.0);

            double w = ip.weight;

            for (int r = 0; r < dim; r++)
                for (int c = 0; c < dim; c++)
                {
                    eps_sum(r, c) += w * eps_local(r, c);
                    sigma_sum(r, c) += w * sigma_local(r, c);
                }
            w_sum += w;
        }

        for (int r = 0; r < dim; r++)
            for (int c = 0; c < dim; c++)
            {
                strain[i](r, c) = eps_sum(r, c) / w_sum;
                stress[i](r, c) = sigma_sum(r, c) / w_sum;
            }
    }
}

// Compute von Mises stress scalar from stress tensor
double VonMisesStress(const DenseMatrix &s)
{
    int dim = s.Height();
    double s_xx = s(0,0);
    double s_yy = s(1,1);
    double s_zz = (dim == 3) ? s(2,2) : 0.0;
    double s_xy = s(0,1);
    double s_yz = (dim == 3) ? s(1,2) : 0.0;
    double s_zx = (dim == 3) ? s(2,0) : 0.0;

    double vonMises = 0.0;
    if (dim == 2)
    {
        vonMises = sqrt(s_xx*s_xx - s_xx*s_yy + s_yy*s_yy + 3.0*s_xy*s_xy);
    }
    else // dim == 3
    {
        vonMises = sqrt(0.5 * (
            (s_xx - s_yy)*(s_xx - s_yy) + (s_yy - s_zz)*(s_yy - s_zz) + (s_zz - s_xx)*(s_zz - s_xx) +
            6*(s_xy*s_xy + s_yz*s_yz + s_zx*s_zx)
        ));
    }
    return vonMises;
}

// Compute statistics (min, max, mean, stddev) for a vector of doubles
void ComputeStats(const std::vector<double> &vals, double &min_val, double &max_val, double &mean, double &stddev)
{
    if (vals.empty())
    {
        min_val = max_val = mean = stddev = 0.0;
        return;
    }

    min_val = *std::min_element(vals.begin(), vals.end());
    max_val = *std::max_element(vals.begin(), vals.end());
    mean = std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();

    double sq_sum = 0.0;
    for (auto v : vals)
    {
        sq_sum += (v - mean) * (v - mean);
    }
    stddev = sqrt(sq_sum / vals.size());
}

// Compute displacement magnitude at dofs
void GetDisplacementMagnitudes(const GridFunction &x, std::vector<double> &disp_mag)
{
    int size = x.Size();
    int dim = x.FESpace() ? x.FESpace()->GetVDim() : 1;
    dim = (dim == 0) ? 1 : dim;

    disp_mag.resize(size / dim);

    for (int i = 0; i < (size / dim); i++)
    {
        double sum_sq = 0.0;
        for (int d = 0; d < dim; d++)
        {
            double val = x(dim * i + d);
            sum_sq += val * val;
        }
        disp_mag[i] = sqrt(sum_sq);
    }
}

// Compute displacement magnitudes at element dofs filtered by region attribute
void GetRegionDisplacementMagnitudes(const Mesh &mesh, FiniteElementSpace &fespace,
                                     const GridFunction &x,
                                     int region_attr,
                                     std::vector<double> &disp_mag)
{
    disp_mag.clear();

    int NE = mesh.GetNE();
    int dim = mesh.Dimension();

    for (int i = 0; i < NE; i++)
    {
        if (mesh.GetAttribute(i) == region_attr)
        {
            Array<int> vdofs;
            fespace.GetElementVDofs(i, vdofs);

            // vdofs contains dim DOFs per node, total size multiple of dim
            for (int j = 0; j < vdofs.Size(); j += dim)
            {
                double sum_sq = 0.0;
                for (int d = 0; d < dim; d++)
                {
                    sum_sq += x(vdofs[j + d]) * x(vdofs[j + d]);
                }
                disp_mag.push_back(sqrt(sum_sq));
            }
        }
    }
}

// Compute von Mises stress values at element centers
void GetVonMisesAtElements(const std::vector<DenseMatrix> &stress,
                           std::vector<double> &vonMises)
{
    int NE = stress.size();
    vonMises.resize(NE);
    for (int i = 0; i < NE; i++)
    {
        vonMises[i] = VonMisesStress(stress[i]);
    }
}

// Compute von Mises stress values filtered by region attribute
void GetRegionVonMises(const Mesh &mesh,
                       const std::vector<DenseMatrix> &stress,
                       int region_attr,
                       std::vector<double> &vonMises_region)
{
    vonMises_region.clear();
    int NE = mesh.GetNE();

    for (int i = 0; i < NE; i++)
    {
        if (mesh.GetAttribute(i) == region_attr)
        {
            vonMises_region.push_back(VonMisesStress(stress[i]));
        }
    }
}

double ComputeTotalElasticEnergy(Mesh &mesh,
                                 const std::vector<DenseMatrix> &strain,
                                 const std::vector<DenseMatrix> &stress)
{
    double total_energy = 0.0;
    int NE = mesh.GetNE();

    for (int i = 0; i < NE; i++)
    {
        double element_volume = mesh.GetElementVolume(i);
        const DenseMatrix &eps = strain[i];
        const DenseMatrix &sig = stress[i];
        double energy_density = 0.0;

        int dim = eps.Height();
        for (int r = 0; r < dim; r++)
            for (int c = 0; c < dim; c++)
                energy_density += eps(r, c) * sig(r, c);

        energy_density *= 0.5;
        total_energy += energy_density * element_volume;
    }

    return total_energy;
}

// Also update region average energy function
double ComputeRegionAverageStrainEnergyDensity(Mesh &mesh,
                                               const std::vector<DenseMatrix> &strain,
                                               const std::vector<DenseMatrix> &stress,
                                               int region_attr)
{
    double total_energy = 0.0;
    double total_volume = 0.0;
    int NE = mesh.GetNE();

    for (int i = 0; i < NE; i++)
    {
        if (mesh.GetAttribute(i) == region_attr)
        {
            double element_volume = mesh.GetElementVolume(i);
            const DenseMatrix &eps = strain[i];
            const DenseMatrix &sig = stress[i];
            double energy_density = 0.0;

            int dim = eps.Height();
            for (int r = 0; r < dim; r++)
                for (int c = 0; c < dim; c++)
                    energy_density += eps(r, c) * sig(r, c);

            energy_density *= 0.5;

            total_energy += energy_density * element_volume;
            total_volume += element_volume;
        }
    }

    if (total_volume > 0.0)
        return total_energy / total_volume;
    else
        return 0.0;
}

double ComputeNormalTraction(const DenseMatrix &stress, const Vector &normal)
{
    // Assumes 'stress' is a 3x3 Cauchy stress tensor
    // and 'normal' is a 3D unit normal vector

    Vector sigma_n(3);
    stress.Mult(normal, sigma_n);   // σ ⋅ n
    return sigma_n * normal;        // nᵗ ⋅ (σ ⋅ n)
}


int main(int argc, char *argv[])
{
    // --- Input options
    const char *mesh_file = "";
    int order = 1;
    const char *device_config = "cpu";
    // Tumor
    double E = 0.005;       // Young's modulus in MPa (5 kPa)
    double nu = 0.48;       // Poisson's ratio near incompressible

    //Bone
//    double E = 15000.0; // 15 GPa = 15000 MPa
//    double nu = 0.3;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
    args.AddOption(&order, "-o", "--order", "Finite element order.");
    args.AddOption(&device_config, "-d", "--device", "Device string.");
    args.AddOption(&E, "-E", "--young", "Young's modulus.");
    args.AddOption(&nu, "-nu", "--poisson", "Poisson's ratio.");
    args.Parse();
    if (!args.Good()) { args.PrintUsage(cout); return 1; }
    args.PrintOptions(cout);

    // --- Device configuration
    Device device(device_config);
    device.Print();

    auto start = std::chrono::high_resolution_clock::now();

    // --- Load and refine mesh
    Mesh mesh(mesh_file, 1, 1);
    PrintMeshAttributes(mesh);

    int dim = mesh.Dimension();
//    mesh.UniformRefinement();

    // --- FE space: Vector-valued H1
    H1_FECollection fec(order, dim);
    FiniteElementSpace fespace(&mesh, &fec, dim);

/*
    // --- Define boundary conditions
    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1;  // Fix boundary attribute 1 (Dirichlet)
    Array<int> ess_tdof_list;
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

*/
    // --- Essential boundary setup (robust)
    int bmax = mesh.bdr_attributes.Max();           // maximum attribute value (e.g. 2)
    Array<int> ess_bdr(bmax);
    ess_bdr = 0;

    // print info for debugging
    cout << "mesh.bdr_attributes.Max() = " << bmax << endl;
    cout << "Boundary attributes present: ";
    for (int i = 1; i <= bmax; ++i)
    {
        // test if attribute 'i' actually exists on the mesh boundary elements
        // (mesh.GetNBE() iterates bdr elements; PrintMeshAttributes() already printed values)
        cout << (i) << " ";
    }
    cout << endl;

    // Mark attribute 2 (your mesh) as essential. If you want to mark attr 1 instead,
    // change the index accordingly to attr-1.
    if (bmax >= 2)
    {
        ess_bdr[2-1] = 1; // mark boundary attribute = 2
    }
    else
    {
        // Fallback: mark attribute 1 if only 1 exists
        ess_bdr[0] = 1;
    }

    Array<int> ess_tdof_list;
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    cout << "Number of essential true DOFs: " << ess_tdof_list.Size() << endl;

//---------------------------------
    // --- Solution GridFunction (declare BEFORE usage)
    GridFunction x(&fespace);
    x = 0.0;

    // --- Define displacement BC on boundary attribute 1 (example: x displacement = -0.05)
    GridFunction disp_bc(&fespace);
    disp_bc = 0.0;

    for (int i = 0; i < mesh.GetNBE(); i++) {
        int bdr_attr = mesh.GetBdrAttribute(i);
        // Apply the prescribed displacement on the loaded boundary attribute (2),
        // to match ess_bdr marking which sets attribute 2 as essential.
        if (bdr_attr == 2) {
            Array<int> vdofs;
            fespace.GetBdrElementVDofs(i, vdofs);  // get boundary dofs for boundary element i
            for (int j = 0; j < vdofs.Size(); j += dim) {
                // Apply displacement in x-direction (component 0)
                disp_bc(vdofs[j]) = -0.00098;
            }
        }

    }

    // Apply essential boundary conditions to solution GridFunction
    for (int i = 0; i < ess_tdof_list.Size(); i++)
    {
        x(ess_tdof_list[i]) = disp_bc(ess_tdof_list[i]);
    }

    // --- Define force vector f = (1,0) or (1,0,0)
    Vector f_vec(dim);
    f_vec = 0.0; // X, Y , Z

 //tumor
    f_vec[0] = 1.03e-5;  // Body force in N/mm³ (density * gravity)

 //bone
 //   f_vec[2] = -1.7658e-5;  // N/mm³ for bone

    VectorConstantCoefficient f_coeff(f_vec);

    LinearForm b(&fespace);
    b.AddDomainIntegrator(new VectorDomainLFIntegrator(f_coeff));
    b.Assemble();

    // --- Define elasticity integrator (bilinear form)
    double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double mu = E / (2.0 * (1.0 + nu));

    ConstantCoefficient lambda_c(lambda);
    ConstantCoefficient mu_c(mu);

    BilinearForm a(&fespace);
    a.AddDomainIntegrator(new ElasticityIntegrator(lambda_c, mu_c));
    a.Assemble();

    // --- Form and solve linear system
    OperatorPtr A;
    Vector B, X;
    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

    cout << "Linear system size: " << A->Height() << endl;

#ifndef MFEM_USE_SUITESPARSE
    GSSmoother M((SparseMatrix&)(*A));
    PCG(*A, M, B, X, 1, 1000, 1e-12, 0.0);
#else
    UMFPackSolver solver;
    solver.SetOperator(*A);
    solver.Mult(B, X);
#endif

    a.RecoverFEMSolution(X, b, x);

//    // --- Form and solve linear system using Hypre AMG preconditioner
//    OperatorPtr A;
//    Vector B, X;
//    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
//
//    cout << "Linear system size: " << A->Height() << endl;
//
//    #ifdef MFEM_HYPRE_VERSION
//        HypreBoomerAMG amg;
//        amg.SetPrintLevel(1);   // show convergence info
//        PCG(*A, amg, B, X, 1, 500, 1e-8, 0.0);
//    #else
//        GSSmoother M((SparseMatrix&)(*A));
//        PCG(*A, M, B, X, 1, 500, 1e-8, 0.0);
//    #endif
//
//    // Recover the solution
//    a.RecoverFEMSolution(X, b, x);


    // --- Compute strain and stress tensors per element
    std::vector<DenseMatrix> strain, stress;
    ComputeStrainStress(fespace, x, lambda, mu, strain, stress);

    // --- Compute von Mises stress and save in GridFunction for visualization
    GridFunction vonMisesStress(&fespace);
    vonMisesStress = 0.0;

    for (int i = 0; i < mesh.GetNE(); i++)
    {
        double vm = VonMisesStress(stress[i]);
        Array<int> vdofs;
        fespace.GetElementVDofs(i, vdofs);
        for (int j = 0; j < vdofs.Size(); j++)
        {
            vonMisesStress(vdofs[j]) = vm;
        }
    }

    // --- Create surface mesh from boundary elements (older MFEM version compatible)
    Mesh *surface_mesh = new Mesh(mesh, true);  // Extracts boundary mesh

    // Build mapping: surface element i → volume element index
    std::vector<int> surf_to_vol_map;
    for (int i = 0; i < mesh.GetNBE(); i++)
    {
        int elem, info;
        mesh.GetBdrElementAdjacentElement(i, elem, info);
        surf_to_vol_map.push_back(elem);
    }
    // --- Set up finite element space on the surface mesh
    int surf_dim = surface_mesh->Dimension();
    H1_FECollection surf_fec(order, surf_dim);
    FiniteElementSpace surf_fes(surface_mesh, &surf_fec);

    // --- Project coordinates and compute Laplacian for curvature estimation
    std::vector<GridFunction*> coords(surf_dim);
    std::vector<GridFunction*> laplacians(surf_dim);
    for (int d = 0; d < surf_dim; d++)
    {
        // Coordinate component d
        FunctionCoefficient coord_d([d](const Vector &x) { return x(d); });

        coords[d] = new GridFunction(&surf_fes);
        coords[d]->ProjectCoefficient(coord_d);

        // Laplace system
        BilinearForm laplace(&surf_fes);
        laplace.AddDomainIntegrator(new DiffusionIntegrator);
        laplace.Assemble();
        laplace.Finalize();

        LinearForm rhs(&surf_fes);
        rhs = 0.0;

        laplacians[d] = new GridFunction(&surf_fes);
        GSSmoother smoother(laplace.SpMat());
        PCG(laplace.SpMat(), smoother, rhs, *laplacians[d], 0, 1000, 1e-12, 0.0);
    }

    // --- Compute mean curvature = ||Laplace(x), Laplace(y), Laplace(z)||
    GridFunction mean_curvature(&surf_fes);
    for (int i = 0; i < surf_fes.GetNDofs(); i++)
    {
        double sum_sq = 0.0;
        for (int d = 0; d < surf_dim; d++)
        {
            double val = (*laplacians[d])(i);
            sum_sq += val * val;
        }
        mean_curvature(i) = sqrt(sum_sq);
    }

    // --- Compute normal traction σₙ and contraction ratio γ = κ * σₙ
    GridFunction gamma(&surf_fes);           // contraction ratio
    GridFunction normal_traction(&surf_fes); // normal traction
    gamma = 0.0;
    normal_traction = 0.0;

    Vector normal(3);
    DenseMatrix stress_2(3);


    cout << "surface_mesh->GetNE(): " << surface_mesh->GetNE() << endl;
    cout << "surf_to_vol_map.size(): " << surf_to_vol_map.size() << endl;

    // Loop over surface elements
    //for (int i = 0; i < surface_mesh->GetNE(); i++)
    for (int i = 0; i < surf_to_vol_map.size(); i++)
    {
        ElementTransformation *T = surface_mesh->GetElementTransformation(i);

        const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(), 1);
        IntegrationPoint ip = ir.IntPoint(0);
        T->SetIntPoint(&ip);

        // Compute surface normal
        CalcOrtho(T->Jacobian(), normal);
        normal /= normal.Norml2();

        // Get stress tensor from corresponding volume element
        int volume_elem = surf_to_vol_map[i];
        if (volume_elem < 0 || volume_elem >= stress.size()) {
            std::cerr << "Invalid volume element index: " << volume_elem << std::endl;
            continue;
        }

        const DenseMatrix &s = stress[volume_elem];
        stress_2 = s;

        // Compute normal traction σₙ = nᵗ σ n
        double sigma_n = ComputeNormalTraction(stress_2, normal);

        // Get surface DOFs
        Array<int> vdofs;
        surf_fes.GetElementVDofs(i, vdofs);

        // Assign values
        for (int j = 0; j < vdofs.Size(); j++)
        {
            double kappa = mean_curvature(vdofs[j]);
            if (fabs(kappa) > 1e-3)  // skip nearly flat regions
            {
                normal_traction(vdofs[j]) = sigma_n;
                gamma(vdofs[j]) = kappa * sigma_n;
            }
        }
    }


    // --- Output full mesh (volume) data
    ParaViewDataCollection dc("Elasticity", &mesh);
    dc.SetPrefixPath("ParaView");
    dc.RegisterField("displacement", &x);
    dc.RegisterField("vonMisesStress", &vonMisesStress);
    dc.SetCycle(0);
    dc.SetTime(0.0);
    dc.Save();

    // --- Output surface mesh data
    ParaViewDataCollection dc_surf("Surface", surface_mesh);
    dc_surf.SetPrefixPath("ParaView");
    dc_surf.RegisterField("mean_curvature", &mean_curvature);
    dc_surf.RegisterField("normal_traction", &normal_traction);
    dc_surf.RegisterField("gamma", &gamma);
    dc_surf.SetCycle(0);
    dc_surf.SetTime(0.0);
    dc_surf.Save();


    cout << "Output saved in 'ParaView/' directory." << endl;

    // --- Additional computations: displacement & stress stats, strain energy, tumor stiffness

    // Displacement magnitude stats (global)
    std::vector<double> disp_global;
    GetDisplacementMagnitudes(x, disp_global);

    double min_disp, max_disp, mean_disp, stddev_disp;
    ComputeStats(disp_global, min_disp, max_disp, mean_disp, stddev_disp);

    cout << "Global displacement magnitude stats:" << endl;
    cout << "  min: " << min_disp << ", max: " << max_disp
         << ", mean: " << mean_disp << ", stddev: " << stddev_disp << endl;

    // Tumor region assumed attribute = 1 (adjust if needed)
    int tumor_attr = 1;

    // Displacement stats in tumor region
    std::vector<double> disp_tumor;
    GetRegionDisplacementMagnitudes(mesh, fespace, x, tumor_attr, disp_tumor);

    ComputeStats(disp_tumor, min_disp, max_disp, mean_disp, stddev_disp);

    cout << "Tumor region displacement magnitude stats (attr=1):" << endl;
    cout << "  min: " << min_disp << ", max: " << max_disp
         << ", mean: " << mean_disp << ", stddev: " << stddev_disp << endl;

    // Von Mises stress stats (global)
    std::vector<double> vonMises_global;
    GetVonMisesAtElements(stress, vonMises_global);

    double min_vm, max_vm, mean_vm, stddev_vm;
    ComputeStats(vonMises_global, min_vm, max_vm, mean_vm, stddev_vm);

    cout << "Global von Mises stress stats:" << endl;
    cout << "  min: " << min_vm << ", max: " << max_vm
         << ", mean: " << mean_vm << ", stddev: " << stddev_vm << endl;

    // Von Mises stress stats in tumor region
    std::vector<double> vonMises_tumor;
    GetRegionVonMises(mesh, stress, tumor_attr, vonMises_tumor);

    ComputeStats(vonMises_tumor, min_vm, max_vm, mean_vm, stddev_vm);

    cout << "Tumor region von Mises stress stats (attr=1):" << endl;
    cout << "  min: " << min_vm << ", max: " << max_vm
         << ", mean: " << mean_vm << ", stddev: " << stddev_vm << endl;

    // Total elastic strain energy
    double total_energy = ComputeTotalElasticEnergy(mesh, strain, stress);
    cout << "Total elastic strain energy: " << total_energy << endl;

    // Tumor stiffness estimate = average strain energy density in tumor region
    double tumor_stiffness = ComputeRegionAverageStrainEnergyDensity(mesh, strain, stress, tumor_attr);
    cout << "Estimated tumor stiffness (avg strain energy density in attr=1): " << tumor_stiffness << endl;
    //---------------------------time----------------------------
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::ofstream logfile("output.log", std::ios::app);
    logfile << "Elapsed time: " << elapsed.count() << " seconds\n";

    // Also print to console if you want
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

    return 0;
}

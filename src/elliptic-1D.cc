// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif
#include <iostream>
#include <dune/common/fvector.hh>
#include <dune/istl/bvector.hh>
#include <dune/common/parallel/mpihelper.hh> // An initializer of MPI
#include <dune/common/fmatrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/preconditioners.hh>

#include <dune/grid/onedgrid.hh>
#include <dune/grid/io/file/vtk.hh>

double f(double x){return std::sin(2*M_PI*x);}

int main(int argc, char** argv)
{
    Dune::MPIHelper::instance(argc, argv);

    const double L = 1.0; //interval (0,1)
    const double g0 = 0.0;
    const double g1 = 2.0;
    const int N = 1000; // broj elemenata: x_0=0,...x_N = L;
    const double h = L/N;

    using Vector = Dune::BlockVector<double>; //Dune::BlockVector<Dune::FiledVector<double,1>
    using Matrix = Dune::BCRSMatrix<double>; //Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1>>;

    Vector F(N+1), U(N+1);
    Matrix A;

    //kreiranje desne strane
    F[0]=g0; F[N]=g1;
    for(int i=1;i<N;++i) F[i] = h*h*f(i*h); // x_i = i*h;

    for(int i=0;i<=N;++i)
        U[i] = 0.0;

    //kreiranje matrice;
    A.setSize(N+1,N+1);
    A.setBuildMode(Matrix::random);
    // koliko je lemenata u svakom pojedinom retku
    A.setrowsize(0,1);
    A.setrowsize(N,1);
    for(int i=1;i<N;++i)
        A.setrowsize(i,3);
    A.endrowsizes();

    A.addindex(0,0);
    A.addindex(N,N);
    // naznačimo koji su aktivni indeksi u matrici
    for(int i=1;i<N;++i){
        A.addindex(i,i);
        A.addindex(i,i-1);
        A.addindex(i,i+1);
    }
    A.endindices();

    A[0][0] = 1.0;A[N][N] = 1.0;
    for(int i=1;i<N;++i){
        A[i][i] = 2.0;
        A[i][i-1] = -1.0;
        A[i][i+1] = -1.0;
    }
    //gotova matrica

    //selekcija solvera
    //ovaj kod od matrice naopravi adapter
    Dune::MatrixAdapter<Matrix,Vector,Vector> op(A);
    Dune::SeqILU<Matrix,Vector,Vector> ilu(A,0,0.92);
    Dune::BiCGSTABSolver<Vector> solver(op,ilu,1e-12,300,5);
    Dune::InverseOperatorResult r;

    Vector FF = F;
    solver.apply(U,F,r);

    if(r.converged){
       std::cout<< "Solver converged.\n";
       std::cout << "No of iterations = " << r.iterations
             << ", reduction = " << r.reduction << std::endl;
     }
     else
        std::cout<< "Solver did not converge.\n";

    Vector Res(N+1); // želimo = Res = F - AU
    op.apply(U,Res); // Res = AU
    Res -= FF;
    std::cout << " Norma reziduala =" <<  Res.two_norm() << "\n";

    Vector Error(N+1);
    for(int i=0;i<=N;++i)
        Error[i] = U[i] - (2*i*h + std::sin(2*M_PI*i*h)/(4*M_PI*M_PI));// 2x +sin(2pix)/(4pi^2)


    std::cout << " Norma greške =" <<  Error.two_norm() << "\n";
    Dune::OneDGrid grid(N,0.0,L);
    using GV = Dune::OneDGrid::LeafGridView;
    Dune::VTKWriter<GV> writer(grid.leafGridView());
    writer.addVertexData(U,"sol");
    writer.write("out");

    return 0;

}

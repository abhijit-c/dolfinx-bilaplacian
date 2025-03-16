import dolfinx
import dolfinx.fem.petsc
import numpy as np
from petsc4py import PETSc
import pyvista as pv
from scipy.sparse import diags_array
from scipy.sparse.linalg import LinearOperator, factorized
import ufl


class Bilaplacian:
    def __init__(
        self,
        V: dolfinx.fem.FunctionSpace,
        gamma: float,
        delta: float,
        mean: dolfinx.fem.Function | PETSc.Vec | None = None,
        robin_bc: bool = False,
    ):
        r"""
        Create an infinite-dimensional Gaussian measure with bi-Laplacian covariance
        operator. That is, covariance given by the operator $C = (\delta I + \gamma
        {\rm div} \nabla)^{-2}$.

        Parameters
        ----------
        V: dolfinx.fem.FunctionSpace
            Finite element discritization of the space
        gamma: float
            Covariance parameter
        delta: float
            Covariance parameter
        mean : dolfinx.fem.Function | PETSc.Vec | None, default: ``0``
            Mean of the distribution.
        robin_bc: bool
            Whether to employ a Robin boundary condition to minimize boundary artifacts.

        Attributes
        ----------
        V: dolfinx.fem.FunctionSpace
            Finite element discritization of the space
        mean : PETSc.Vec, default: ``0``
            Mean of the distribution.
        A: PETSc.Mat
            Discretization of bi-Laplacian operator
        Ainv: PETSc.KSP
            Facotrized linear operator representing $A^{-1}$.
        M: PETSc.Mat
            Discretization of underling mass matrix
        Minv: PETSc.Mat
            Operator representing $M^{-1}$.
        sqrtM: PETSc.Mat
            Matrix square root of M
        sqrtMinv: PETSc.Mat
            Inverse for matrix square root of M

        Methods
        -------
        logpdf(x)
            Evaluate $||x-x_0||_{R^{-1}}$ where $x_0$ is the mean.
        grad_logpdf(x)
            Evaluate $R^{-1} (x-x_0)$ where $x_0$ is the mean.
        R: PETSc.KSP
            Operator for the underlying covariance matrix.
        Rinv: PETSc.KSP
            Operator for the underlying precision matrix

        """
        self._V = V

        trial = ufl.TrialFunction(V)
        test = ufl.TestFunction(V)

        bilaplacian_varf = (
            gamma * ufl.inner(ufl.grad(trial), ufl.grad(test))
            + delta * ufl.inner(trial, test)
        ) * ufl.dx
        if robin_bc:
            robin_coef = gamma * np.sqrt(delta / gamma)
            bilaplacian_varf += robin_coef * ufl.inner(trial, test) * ufl.ds
        self.A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(bilaplacian_varf))
        self.A.assemble()

        self.A_solver = PETSc.KSP().create(self.V.mesh.comm)
        self.A_solver.setType(PETSc.KSP.Type.PREONLY)
        self.A_solver.getPC().setType(PETSc.PC.Type.LU)
        self.A_solver.setOperators(self.A)

        dx_lumped = ufl.Measure("dx", metadata={"quadrature_rule": "vertex"})
        mass_varf = ufl.inner(trial, test) * dx_lumped
        self.M = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(mass_varf))
        self.M.assemble()

        Minv_diag = self.M.getDiagonal().copy()
        Minv_diag.reciprocal()
        self.Minv = PETSc.Mat().createDiagonal(Minv_diag)

        sqrtM_diag = self.M.getDiagonal().copy()
        sqrtM_diag.sqrtabs()
        self.sqrtM = PETSc.Mat().createDiagonal(sqrtM_diag)

        sqrtMinv_diag = sqrtM_diag.copy()
        sqrtMinv_diag.reciprocal()
        self.sqrtMinv = PETSc.Mat().createDiagonal(sqrtMinv_diag)

        if mean is not None:
            self._mean = mean
        else:
            self._mean = self.M.createVecLeft()
            self._mean.zeroEntries()

        self._helper1 = self.M.createVecLeft()
        self._helper2 = self.M.createVecLeft()
        self._helper3 = self.M.createVecLeft()

    def _zero_helpers(self):
        self._helper1.zeroEntries()
        self._helper2.zeroEntries()
        self._helper3.zeroEntries()

    def R(self, x: PETSc.Vec, out: PETSc.Vec | None = None) -> PETSc.Vec:
        self._zero_helpers()
        if out is None:
            out = self.M.createVecLeft()
        self.A.mult(x, self._helper1)
        self.Minv.mult(self._helper1, self._helper2)
        self.A.mult(self._helper2, out)
        return out

    def Rinv(self, x: PETSc.Vec, out: PETSc.Vec | None = None) -> PETSc.Vec:
        self._zero_helpers()
        if out is None:
            out = self.M.createVecLeft()
        self.A_solver.solve(x, self._helper1)
        self.M.mult(self._helper1, self._helper2)
        self.A_solver.solve(self._helper2, out)
        return out

    def logpdf(self, x: PETSc.Vec) -> float:
        r"""
        Evaluate the "logpdf" of the distribution.

        Note: An infinite dimensional distribution does not admit a pdf in this manner.
        However, this is simply a notational convinience to represent a similar
        computation.
        """
        self._zero_helpers()
        self._helper1.waxpy(-1.0, self.mean, x)
        self.R(self._helper1, out=self._helper2)
        return 0.5 * self._helper2.dot(self._helper1)

    def grad_logpdf(self, x: PETSc.Vec, out: PETSc.Vec | None = None) -> PETSc.Vec:
        r"""
        Evaluate the gradient "logpdf" of the distribution.

        Note: An infinite dimensional distribution does not admit a pdf in this manner.
        However, this is simply a notational convinience to represent a similar
        computation.
        """
        self._zero_helpers()
        if out is None:
            out = self.M.createVecLeft()
        self._helper1.waxpy(-1.0, self.mean, x)
        self.R(self._helper1, out=out)
        return out

    def sample(self, white_noise: PETSc.Vec, out: PETSc.Vec | None = None) -> PETSc.Vec:
        r"""
        Get a random sample from the underlying distribution.
        """
        self._zero_helpers()
        if out is None:
            out = self.M.createVecLeft()
        self.A_solver.solve(white_noise, self._helper1)
        self.sqrtM.mult(self._helper1, self._helper2)
        out.waxpy(1.0, self.mean, self._helper2)
        return out

    def _createLUSolver(self) -> PETSc.KSP:
        ksp = PETSc.KSP().create(self.V.mesh.comm)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        ksp.getPC().setType(PETSc.PC.Type.LU)
        return ksp

    @property
    def V(self) -> dolfinx.fem.FunctionSpace:
        """
        Get the underyling basis object for the measure.
        """
        return self._V

    @property
    def mean(self) -> PETSc.Vec:
        """
        Get the underyling basis object for the measure.
        """
        return self._mean


if __name__ == "__main__":
    from mpi4py import MPI

    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 64, 64)
    V = dolfinx.fem.functionspace(domain, ("P", 1))
    N_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    G_pr = Bilaplacian(V, 1.0, 1.0, robin_bc=True)

    np.random.seed(1)
    white_noise_np = [np.random.randn(N_dofs) for _ in range(3)]
    white_noise_petsc = [PETSc.Vec().createWithArray(wn) for wn in white_noise_np]
    samples = [G_pr.sample(wn) for wn in white_noise_petsc]
    vmax = max([np.max(s.getArray()) for s in samples])
    vmin = min([np.min(s.getArray()) for s in samples])

    pl = pv.Plotter(shape=(1, 3), off_screen=True)

    for i, s in enumerate(samples):
        pl.subplot(0, i)
        topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        pl.add_mesh(grid, scalars=s.array, clim=[vmin, vmax], cmap="inferno")
        pl.camera_position = "xy"  # Set camera to a top down view.
        pl.enable_parallel_projection()  # removes perspective distortion
        pl.remove_scalar_bar()  # cleaner view
        pl.background_color = "white"  # set background color to white for transparency.
        pl.zoom_camera(3)

    pl.link_views()  # link the camera views together.
    pl.screenshot(
        filename="samples.png",
        scale=2,
        transparent_background=True,
        window_size=(1800, 600),
    )

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
        mean: dolfinx.fem.Function | None = None,
        robin_bc: bool = False,
    ):
        r"""
        Create an infinite-dimensional Gaussian measure with bi-Laplacian covariance
        operator. That is, covariance given by the operator $C = (\delta I + \gamma
        {\rm div} \nabla)^{-2}$.

        Parameters
        ----------
        V: Basis
            Finite element discritization of the space
        gamma: float
            Covariance parameter
        delta: float
            Covariance parameter
        mean : ArrayLike, default: ``0``
            Mean of the distribution.
        robin_bc: bool
            Whether to employ a Robin boundary condition to minimize boundary artifacts.

        Attributes
        ----------
        V: dolfinx.fem.FunctionSpace
            Finite element discritization of the space
        mean : dolfinx.fem.Function, default: ``0``
            Mean of the distribution.
        R:
            Operator for the underlying covariance matrix.
        Rinv:
            Operator for the underlying precision matrix
        A:
            Discretization of bi-Laplacian operator
        Ainv:
            Factorized linear operator representing $A^{-1}$.
        M:
            Discretization of underling mass matrix
        Minv:
            Operator representing $M^{-1}$.
        sqrtM:
            Matrix square root of M
        sqrtMinv:
            Inverse for matrix square root of M

        Methods
        -------
        logpdf(x)
            Evaluate $||x-x_0||_{R^{-1}}$ where $x_0$ is the mean.
        grad_logpdf(x)
            Evaluate $R^{-1} (x-x_0)$ where $x_0$ is the mean.
        rvs(size=1)
            Sample ``size`` samples from the measure.

        """
        self._V = V
        if mean is not None:
            self._mean = mean
        else:
            self._mean = dolfinx.fem.Function(V).x.array

        trial = ufl.TrialFunction(V)
        test = ufl.TestFunction(V)

        bilaplacian_varf = (
            gamma * ufl.inner(ufl.grad(trial), ufl.grad(test))
            + delta * ufl.inner(trial, test)
        ) * ufl.dx
        if robin_bc:
            robin_coef = gamma * np.sqrt(delta / gamma)
            bilaplacian_varf += robin_coef * ufl.inner(trial, test) * ufl.ds
        self.A = dolfinx.fem.assemble_matrix(
            dolfinx.fem.form(bilaplacian_varf)
        ).to_scipy()
        self.Ainv = LinearOperator(
            dtype=np.float64, shape=self.A.shape, matvec=factorized(self.A)
        )

        dx_lumped = ufl.Measure("dx", metadata={"quadrature_rule": "vertex"})
        mass_varf = ufl.inner(trial, test) * dx_lumped
        self.M = dolfinx.fem.assemble_matrix(dolfinx.fem.form(mass_varf)).to_scipy()
        self.Minv = diags_array(1 / self.M.diagonal())
        self.sqrtM = diags_array(np.sqrt(self.M.diagonal()))
        self.sqrtMinv = diags_array(1 / self.sqrtM.diagonal())

        def R(x: np.ndarray) -> np.ndarray:
            return self.A @ (self.Minv @ (self.A @ x))

        self.R = LinearOperator(dtype=np.float64, shape=self.M.shape, matvec=R)

        def Rinv(x: np.ndarray) -> np.ndarray:
            return self.Ainv @ (self.M @ (self.Ainv @ x))

        self.Rinv = LinearOperator(dtype=np.float64, shape=self.M.shape, matvec=Rinv)

    def logpdf(self, x: np.ndarray) -> float:
        r"""
        Evaluate the "logpdf" of the distribution.

        Note: An infinite dimensional distribution does not admit a pdf in this manner.
        However, this is simply a notational convinience to represent a similar
        computation.
        """
        innov = x - self.mean
        return 0.5 * np.inner(innov, self.R @ innov)

    def grad_logpdf(self, x: np.ndarray) -> np.ndarray:
        r"""
        Evaluate the gradient "logpdf" of the distribution.

        Note: An infinite dimensional distribution does not admit a pdf in this manner.
        However, this is simply a notational convinience to represent a similar
        computation.
        """
        innov = x - self.mean
        return self.R @ innov

    def sample(self, white_noise: np.ndarray) -> np.ndarray:
        r"""
        Get a random sample from the underlying distribution.
        """
        return self.mean + self.sqrtM @ (self.Ainv @ white_noise)

    def rvs(self, size: int = 1) -> np.ndarray:
        """
        Get ``size`` random samples from the underlying distribution.
        """
        if size == 1:
            return self.sample()
        return np.array([self.sample() for _ in range(size)])

    @property
    def V(self) -> dolfinx.fem.FunctionSpace:
        """
        Get the underyling basis object for the measure.
        """
        return self._V

    @property
    def mean(self) -> np.ndarray:
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

    samples = [G_pr.sample(np.random.randn(N_dofs)) for _ in range(3)]
    vmax, vmin = samples.max(), samples.min()

    pl = pv.Plotter(shape=(1, 3))

    for i, s in enumerate(samples):
        pl.subplot(0, i)
        pl.add_mesh(V, scalars=s, shading="gouraud", clim=[vmin, vmax])
        pl.camera_position = "xy"  # Set camera to a top down view.
        pl.enable_parallel_projection()  # removes perspective distortion
        pl.remove_axes()  # cleaner view
        pl.background_color = "white"  # set background color to white for transparency.

    pl.link_views()  # link the camera views together.
    pl.show(screenshot="samples.png", window_size=(1800, 600))

import numpy as np
import sympy as sp
import scipy.sparse as sparse
from math import floor

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2)

    def create_mesh(self, N: int) -> None:
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xij, self.yij ...
        self.N = N
        self.xij, self.yij = np.meshgrid(np.linspace(0, self.L, N+1), np.linspace(0, self.L, N+1), indexing='ij')
        self.h = self.L/N
        
    def D2(self) -> sparse.csr_matrix:
        """Return second order differentiation matrix"""
        N = self.N
        D = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(N + 1, N + 1), format='lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = D[0, :4][::-1]
        return D
        
        
    def laplace(self) -> sparse.csr_matrix:
        """Return vectorized Laplace operator"""
        D2x = 1/self.h**2 * self.D2()
        D2y = 1/self.h**2 * self.D2()
        N = self.N
        return sparse.kron(D2x, sparse.eye(N+1)) + sparse.kron(sparse.eye(N+1), D2y)
        
    def get_boundary_indices(self) -> np.ndarray:
        """Return indices of vectorized matrix that belongs to the boundary"""
        N = self.N
        top = np.array([i for i in range(N+1)])
        bottom = top + (N+1) * N
        left = np.array([i*(N+1) for i in range(1, N)])
        right = left + N
        
        indices = np.concatenate([top, bottom, left, right])
        return indices
    
    def assemble(self) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Return assembled matrix A and right hand side vector b"""
        # return A, b
        A = self.laplace().tolil()
        B = self.get_boundary_indices()
        A[B] = 0
        A[B, B] = 1
        
        b  = sp.lambdify([x, y],  self.f)(self.xij, self.yij).flatten()
        ue = sp.lambdify([x, y], self.ue)(self.xij, self.yij).flatten()
        b[B] = ue[B]
        return A.tocsr(), b
        
    def l2_error(self, u) -> float:
        """Return l2-error norm"""
        ue = sp.lambdify([x, y], self.ue)(self.xij, self.yij)
        return np.sqrt(np.sum((u - ue)**2) * self.h**2)

    def __call__(self, N) -> np.ndarray:
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A, b.flatten()).reshape((N+1, N+1))
        return self.U

    def convergence_rates(self, m=6) -> tuple[list, np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

    def eval(self, x, y):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        i = floor(x/self.h)
        j = floor(y/self.h)
        
        x_low = i*self.h
        y_low = j*self.h
        
        x_high = x_low + self.h
        y_high = y_low + self.h
        
        x_array = [x_high - x, x - x_low]
        y_array = [y_high - y, y - y_low]
        
        U = self.U
        bl = U[i, j]
        br = U[i, j+1]
        tl = U[i+1, j]
        tr = U[i+1, j+1]
        box = np.array([[bl, br], [tl, tr]])
        
        
        return x_array @ box @ y_array / self.h**2
        
        
    
def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h/2, y: 1-sol.h/2}).n()) < 1e-3

if __name__ == '__main__':
    test_convergence_poisson2d()
    test_interpolation()
    print('All tests passed')

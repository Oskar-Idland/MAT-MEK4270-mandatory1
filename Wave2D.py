import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from os import system as sys

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N: int, sparse: bool = False) -> None:
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xji, self.yij = ...
        self.xij, self.yij = np.meshgrid(np.linspace(0, 1, N+1), np.linspace(0, 1, N+1), indexing='ij', sparse=sparse)
        
    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(N + 1, N + 1), format='lil')
        D[0] = 0
        D[-1] = 0
        return D/ self.h**2

    @property
    def w(self) -> float:
        """Return the dispersion coefficient"""
        return self.c*sp.sqrt(self.kx**2 + self.ky**2)

    def ue(self, mx: float, my: float) -> sp.Expr:
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N: int, mx: float, my: float) -> None:
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.U_prev = sp.lambdify((x, y), self.ue(mx, my).subs(t, 0))(self.xij, self.yij)
        self.U = self.U_prev + 1/2 * (self.c * self.dt)**2 * (self.D @ self.U_prev + self.U_prev @ self.D.T)
        
    @property
    def dt(self) -> float:
        """Return the time step"""
        return self.cfl * self.h / self.c

    def l2_error(self, u, t0) -> float:
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        ue = sp.lambdify((x, y), self.ue(self.mx, self.my).subs(t, t0), 'numpy')(self.xij, self.yij)
        return np.sqrt(np.sum((u - ue)**2) * self.h**2)

    def apply_bcs(self):
        """Apply boundary conditions"""
        # Boundary conditions applied in self.D2()
        ...

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.N = N
        self.h = 1/N
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my
        self.kx = mx*np.pi
        self.ky = my*np.pi
        
        self.create_mesh(N)
        self.D = self.D2(N)
        
        self.initialize(N, mx, my)
        
        U_t = np.zeros((Nt, N+1, N+1))
        U_t[0] = self.U_prev
        for n in range(1, Nt): 
            self.U_next = 2*self.U - self.U_prev + (c*self.dt)**2 * (self.D @ self.U + self.U @ self.D.T)
            self.U_prev = self.U
            self.U = self.U_next
            U_t[n] = self.U_next
            
        if store_data > 0:
            return {n: U_t[n] for n in range(0, Nt, store_data)}
        
        else:
            l2_err = self.l2_error(self.U_next, Nt*self.dt)
            return self.h, l2_err   
        
            
        

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

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
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err)
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N: int) -> sparse.lil_matrix:
        D = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(N + 1, N + 1), format='lil')
        D[0, :2] = -2, 2
        D[-1, -2:] = 2, -2
        D /= self.h**2
        return D
        

    def ue(self, mx, my):
        kx = mx*sp.pi
        ky = my*sp.pi
        return sp.cos(kx*x)*sp.cos(ky*y)*sp.cos(self.w*t)

    def apply_bcs(self):
        # Boundary conditions applied in self.D2()
        ...

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    sol = Wave2D()
    solN = Wave2D_Neumann()
    
    tol = 1e-12
    
    cfl = 1/np.sqrt(2)
    Nt = 64 
    N = 64
    
    mx = 3
    my = 3
    
    _, error_dirch = sol(N, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
    _, error_neumann = solN(N, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
    
    assert error_dirch < tol
    assert error_neumann < tol

def create_gif(bound_cond: str, filename: str) -> None:
    '''
    Creates a gif of the wave equation solution
    
    Parameters
    ----------
    bound_cond : str
        The boundary condition to use. Either 'dirchlet' or 'neumann'
    filename : str
        The name of the gif file to create. No file extension needed.
    '''
    
    N = 256
    Nt = 175
    cfl = 1/np.sqrt(2)
    mx = 3
    my = 3
    
    if bound_cond == 'dirchlet':
        sol = Wave2D()
        
    elif bound_cond == 'neumann':
        sol = Wave2D_Neumann()
        
    data = sol(N, Nt, cfl=cfl, mx=mx, my=my, store_data=1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    frames = []
    for n, u in data.items():
        if n % 2 == 0: 
            continue
        
        frame = ax.plot_surface(sol.xij, sol.yij, u, vmin=-0.5*data[0].max(), vmax=data[0].max(), cmap=cm.coolwarm, linewidth=0, antialiased=False)
        frames.append([frame])

    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)
    ani.save(f'{filename}.gif', writer='pillow', fps=30) 
        
    

if __name__ == "__main__":
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d()
    print("All tests passed")
    create_gif(bound_cond='dirichlet', filename='dirchlet')
    create_gif(bound_cond='neumann', filename='neumannwave')

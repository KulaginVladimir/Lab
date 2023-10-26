import festim as F
import fenics as f
import numpy as np
import sympy as sp
from scipy import special

def thermal_cond_function(T):
    return 149.441-45.466e-3*T+13.193e-6*T**2-1.484e-9*T**3+3.866e6/(T+1.e-4)**2
def heat_capacity_function(T):
    return (21.868372+8.068661e-3*T-3.756196e-6*T**2+1.075862e-9*T**3+1.406637e4/(T+1.)**2) / 183.84e-3   

def q_tot(t):
    tau = 250e-6
    q_max = 10e6
    return q_max*sp.exp(-(tau/t)**2)
def cooling(T, mobile):
    return -1.98e7 * (3.35e-3 * (T-373) + 1.2e-1 * (1 - f.exp(-6.9e-2*(T-373))))
def flux(t):
    q = 1.6e-19 #C
    q_max = 10e6
    E = 115
    return q_max/(E+13.6)/q*sp.exp(-(tau/t)**2)

w_atom_density = 6.31e28  # atom/m3

my_model = F.Simulation()

vertices = np.concatenate([
    np.linspace(0, 5e-8, num=500),
    np.linspace(5e-8, 1e-5, num=200),
    np.linspace(1e-5, 1e-4, num=50),
    np.linspace(1e-4, 6e-3, num=20),])

my_model.mesh = F.MeshFromVertices(vertices=vertices)

tungsten = F.Material(id=1, 
                    D_0=1.97e-7, 
                    E_D=0.2,
                    thermal_cond=thermal_cond_function, 
                    heat_capacity=heat_capacity_function,
                    rho=19250,
                    H={"free_enthalpy": -0.0045, "entropy": 0})

my_model.materials = tungsten

trap = F.Trap(
        k_0=1.97e-7/(1.1e-10**2*w_atom_density),
        E_k=0.2,
        p_0=1e13,
        E_p=1.5,
        density=1e-5*w_atom_density,
        materials=tungsten
    )

my_model.traps = [trap]

impl_source = F.ImplantationFlux(
    flux=flux(F.t),  # H/m2/s
    imp_depth=200e-10,  # m
    width=15e-10,  # m
    volume=1
)

my_model.sources = [impl_source]

my_model.boundary_conditions = [
    F.FluxBC(value=q_tot(F.t), field="T", surfaces=1),
    F.CustomFlux(function=cooling, field="T", surfaces=2),
    F.DirichletBC(surfaces=[1,2],value=0, field="solute")
]

absolute_tolerance = 1e-4
relative_tolerance = 1e-6

my_model.T = F.HeatTransferProblem(
    absolute_tolerance=absolute_tolerance,
    relative_tolerance=relative_tolerance,
    transient=True,
    initial_value=373,
    maximum_iterations=200
)

my_model.dt = F.Stepsize(
    initial_value=1e-5,
    stepsize_change_ratio=1.01,
    t_stop = 0.,
    stepsize_stop_max = 1e-4,
    dt_min=1e-7
)

my_model.settings = F.Settings(
    absolute_tolerance=absolute_tolerance,
    relative_tolerance=relative_tolerance,
    transient=True,
    final_time=100,
    soret=True,
)

results_folder = "./"
derived_quantities = F.DerivedQuantities([F.HydrogenFlux(surface=1), 
                                          F.TotalSurface(field='T', surface=1), 
                                          F.TotalVolume(field='retention', volume=1),
                                          F.TotalVolume(field='solute', volume=1),
                                          F.TotalVolume(field="1", volume=1)],
                                         nb_iterations_between_compute = 10,
                                         filename=results_folder + "derived_quantities_"+str(sys.argv[1])+".csv")
my_model.exports = [derived_quantities]
my_model.initialise()
my_model.run()

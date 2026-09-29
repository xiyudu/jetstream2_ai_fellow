import numpy as onp

from jax import config
config.update('jax_enable_x64', True)

import jax.numpy as jnp
from jax import random, jit, lax, vmap, jacfwd, remat, value_and_grad
from jax.example_libraries import optimizers

import time
import os
from jax_md import space, smap, energy, minimize, quantity, simulate, rigid_body

def get_psi_k(displacement_all, k=6, r0=1.1, alpha=100):

  def weight(r):
    return jnp.where(r<1e-7, 0., 1.0/(1 + jnp.exp(alpha*(r - r0))))

  i_imaginary = complex(0,1)
  def get_ylms(dR_ij):
    epsilon = 0.00001 #avoids nan in derivative
    dR_ij = jnp.where(dR_ij==0.0, epsilon, dR_ij)
    theta = jnp.arctan2(dR_ij[1], dR_ij[0])
    return jnp.exp(i_imaginary*k*theta)

  def calculate_order_param(R):
    v_get_ylms = vmap(vmap(get_ylms))
    ds = displacement_all(R, R)
    r = space.distance(ds)
    w = weight(r)
    psi_k = (jnp.sum(v_get_ylms(ds)*w, axis=0)/(jnp.sum(w, axis=0)+0.00001))
    return psi_k

  def average_order_param(R):
    psi_k = calculate_order_param(R)
    return jnp.abs(jnp.mean(psi_k))

  return average_order_param

def make_2D_lattice_from_index_list(index_list,
                                    a0=jnp.array([jnp.sqrt(2),0.0]),
                                    a1=jnp.array([0.0,jnp.sqrt(2)]),
                                    unitcell=jnp.array([[0.0, 0.0]]),
                                    offset=jnp.array([0.0,0.0])):
  unitcell_particles = unitcell + offset 
  N = len(unitcell_particles)*index_list.shape[0]
  def get_cell(i,j):
    cell_shift = i*a0+j*a1
    return unitcell_particles + cell_shift

  return jnp.reshape(vmap(get_cell, in_axes=(0, 0))(index_list[:,0],index_list[:,1]),(N,2))

def make_2D_lattice(irange, jrange, **kwargs):
  index_list = jnp.array([ [i,j] for i in irange for j in jrange])
  return make_2D_lattice_from_index_list(index_list, **kwargs)

make_2D_lattice = jit(make_2D_lattice)

@jit
def thetas_to_shape(theta, species, radius=0.5, center_mass=1.0, patch_mass=1e-5):
# def thetas_to_shape(thetas, species, radius=0.5, center_mass=1.0, patch_mass=1e-5):
  # Convert angles to patch positions
  #thetas = thetas.at[0].set(0.0)
  thetas = jnp.array([0, theta, -theta, jnp.pi-theta, jnp.pi, jnp.pi+theta])
  patch_positions = jnp.zeros((len(thetas), 2), dtype = jnp.float64)
  patch_positions = patch_positions.at[:,0].set(radius*jnp.cos(thetas))
  patch_positions = patch_positions.at[:,1].set(radius*jnp.sin(thetas))

  # Add the central particle (position (0, 0)) to the list of positions
  central_particle_position = jnp.array([[0.0, 0.0]])
  positions = jnp.concatenate((central_particle_position, patch_positions), axis=0)

  if not len(thetas)+1 == len(species):
    raise ValueError("Must provide species information for every particle")

  # Combine positions and species to form a patchy particle rigid_body object
  masses = jnp.array([center_mass] + len(thetas)*[patch_mass])
  shape = rigid_body.point_union_shape(positions, masses).set(point_species=species)
  return shape

@jit
def energy_matrix(eng):
  i, j =jnp.triu_indices(NUM_SPECIES)
  eng_m = jnp.zeros((NUM_SPECIES, NUM_SPECIES))
  eng_m = eng_m.at[i, j].set(eng)
  return eng_m.at[j, i].set(eng)

### Global Parameters
M = 6
A = 1.5
A0_sq = jnp.array([A, 0])
A1_sq = jnp.array([0, A])
UNITCELL_sq = jnp.array([[0.0,0.0]])
N = M**2
DIMENSION = 2

RHO = 0.3
RADIUS = 0.5
BOX_SIZE = (N/RHO)**(1/2)

kT = 1.0
dt = 1e-3

K = 6
R0 = RADIUS*2*1.1
ALPHA = 100

THETA = 30*jnp.pi/180.0
# THETAS = jnp.array([0, jnp.pi/2.0, jnp.pi, jnp.pi*3/2.0])
# NUM_PATCHES = len(THETAS)
NUM_PATCHES = 6
SPECIES = jnp.array([0, 1, 1, 1, 1, 1, 1], dtype = jnp.int32)
NUM_SPECIES = len(set(list(onp.array(SPECIES))))
ENERGIES = jnp.array([0, 0, 4.0])
# ENERGIES = jnp.array([0, 0, 0, 0, 0, 
#                       0, 0, 4.0, 0, 0, 
#                       0, 4.0, 0, 0, 0])
# THETAS_AND_ENERGIES = jnp.concatenate([THETAS, ENERGIES])

NUM_STEPS_TO_RUN=40000
NUM_STEPS_TO_OPT=10000
INNER_STEPS = 1000
OPT_STEPS = 300
BATCH_SIZE = 1
SAVE_EVERY = 1

key=random.PRNGKey(128)

def run_sim(thetas, initial_positions, num_steps, key, kT):
#def run_sim(thetas_and_energies, initial_positions, num_steps, key, kT):

  displacement, shift = space.periodic(BOX_SIZE)
  key, split = random.split(key)

  # Setup Rigid Bodies
  # thetas = thetas_and_energies[:NUM_PATCHES]
  patchy_particle_shape = thetas_to_shape(thetas, SPECIES)
  angle_key, split = random.split(key)
  orientations = random.uniform(angle_key, (len(initial_positions),), dtype=jnp.float64) * jnp.pi * 2
  full_configuration = rigid_body.RigidBody(initial_positions, orientations)

  # Setup Interactions
  soft_eps = jnp.zeros((NUM_SPECIES, NUM_SPECIES))
  soft_eps = soft_eps.at[0, 0].set(10000.0)
  morse_eps = energy_matrix(ENERGIES)
  # morse_eps = energy_matrix(thetas_and_energies[NUM_PATCHES:])

  energy_fn_soft = energy.soft_sphere_pair(displacement, species = NUM_SPECIES, sigma = RADIUS*2, epsilon = soft_eps, alpha = 4.0)
  energy_fn_morse = energy.morse_pair(displacement, species = NUM_SPECIES, sigma = 0.0, epsilon = morse_eps, alpha = 9.0, r_cutoff = 1.5)
  energy_fn_all = lambda R, **kwargs: energy_fn_soft(R, **kwargs) + energy_fn_morse(R, **kwargs)
  energy_fn = rigid_body.point_energy(energy_fn_all, patchy_particle_shape)
  energy_fn = jit(energy_fn)

  # Setup Integrator
  init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT = kT)
  state = init_fn(key, full_configuration, mass=patchy_particle_shape.mass())

  do_step = lambda state, t: (step_fn(state), 0.)
  do_step = jit(do_step)
  inner_steps = jnp.arange(INNER_STEPS)

  @remat
  def do_outer_step(state, i):
    state, _ = lax.scan(do_step, state, inner_steps)
    return state, 0.

  do_outer_step = jit(do_outer_step)
      
  steps = jnp.arange(int(num_steps/INNER_STEPS))
  state, losses = lax.scan(do_outer_step, state, steps)

  return state

run_sim = jit(run_sim, static_argnums = 2)
v_run_sim = jit(vmap(run_sim, in_axes = (None, None, None, 0, None)), static_argnums = 2)
many_states_run_sim = jit(vmap(run_sim, in_axes=(None, 0, None, 0, None)), static_argnums = 2)

@jit
def loss_fn(R, k=6, r0=1.1, alpha=100):
  displacement, shift = space.periodic(BOX_SIZE)
  displacement_all = vmap(vmap(displacement, (0, None), 0), (None, 0), 0)
  order_param = get_psi_k(displacement_all, k = K, r0 = R0, alpha = ALPHA)
  return (1-order_param(R))**2

v_loss = vmap(loss_fn, in_axes = (0))
v_loss = jit(v_loss)
avg_loss = lambda R_batched: jnp.mean(v_loss(R_batched))
avg_loss = jit(avg_loss)

def run_partial_sim(params, key, batch_size):
  R = (make_2D_lattice)(jnp.arange(0,M),jnp.arange(0,M),
                        a0 = A0_sq,
                        a1 = A1_sq,
                        unitcell= UNITCELL_sq,
                        offset=jnp.array([0.01, 0.03]))
  sim_keys = random.split(key, batch_size)
  states = v_run_sim(params, R, NUM_STEPS_TO_RUN, sim_keys, kT)
  return states.position.center

def get_mean_loss(params, initial_positions, keys):
  # pdb.set_trace()
  states = many_states_run_sim(params, initial_positions, NUM_STEPS_TO_OPT, keys, kT)
  return avg_loss(states.position.center), states.position

g_mean_loss = jit(value_and_grad(get_mean_loss, has_aux=True))

def optimize(input_params, key, opt_steps, batch_size, save_every, filename, learning_rate=0.01):

  learning_rate_schedule = jnp.ones(opt_steps)*learning_rate
  ind = int(opt_steps / 3)
  learning_rate_schedule = learning_rate_schedule.at[ind:2*ind].set(learning_rate * 0.5)
  learning_rate_schedule = learning_rate_schedule.at[2*ind:].set(learning_rate * 0.1)
  learning_rate_fn = lambda i: learning_rate_schedule[i]

  opt_init, opt_update, get_params = optimizers.adam(step_size=learning_rate_fn)

  def clip_gradient(g, clip=10000.0):
    return jnp.array(jnp.where(jnp.abs(g) > clip, jnp.sign(g)*clip, g))

  def step(i, opt_state, key, batch_size=10, save_every=10):

    params = get_params(opt_state)
    key, split = random.split(key)
    simulation_keys = random.split(split, batch_size)
    initial_positions = run_partial_sim(params, key, batch_size)
      
    vals, gs = g_mean_loss(params, initial_positions, simulation_keys)
    loss, final_positions = vals
    gs = clip_gradient(gs)
    g = gs
    # gs = vmap(clip_gradient)(gs)

    # if BATCH_SIZE > 1:
    #   g = jnp.mean(jnp.array(gs), axis = 0)
    # else:
    #   g = gs

    if(i%save_every==0):

      print("Loss: {}".format(loss))
      print("Parameters: {}".format(params))
      print("Gradient: {}".format(g))

    return opt_update(i, g, opt_state), loss, g, final_positions

  opt_state = opt_init(input_params)
  iteration = 0
  i0 = iteration
  min_loss_params = input_params
  min_loss = 1e6
  losses = []
  gradients = []
  opt_params = []
  opt_traj = []
  for _ in range(i0,i0 + opt_steps):
    iteration += 1
    key, split = random.split(key)
    new_opt_state, loss, g, final_positions = step(iteration, opt_state, split, batch_size=batch_size, save_every=save_every)
    if loss < min_loss:
      min_loss = loss
      min_loss_params = get_params(opt_state)
    opt_state = new_opt_state
    losses.append(loss)
    gradients.append(g)
    opt_params.append(get_params(opt_state))
    opt_traj.append(final_positions)

  return min_loss, min_loss_params, losses, gradients, opt_params, opt_traj

min_loss, min_loss_params, losses, gradients, opt_params, opt_traj = optimize(THETA, key, OPT_STEPS, 
                                                                     BATCH_SIZE, SAVE_EVERY, '', learning_rate = 0.1)
print('Min loss: ', min_loss)
print('Min loss params: ', min_loss_params)
onp.save("losses_kagome.npy", losses)
onp.save("gradients_kagome.npy", gradients)
onp.save("params_kagome.npy", opt_params)
onp.save("trajectories_kagome.npy", opt_traj)

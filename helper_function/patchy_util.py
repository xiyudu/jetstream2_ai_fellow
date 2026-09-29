### This file has a collection of util functions for patchy particles
### Such as patchy particle definitions and visualizations

### thetas_to_shape shows how to use angle informations to build a patchy particle using
### jax-md's rigid_body classes
### there are many ways to define a patchy particle, one may use one single theta value 
### and then populate angles for each patches or explicitly define everything
### this definition directly controls what parameters are being optimized so choose the 
### style of definition as one see fit
@jit
def thetas_to_shape_2D(theta, species, radius=0.5, center_mass=1.0, patch_mass=1e-5):
  # Convert angles to patch positions
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
def thetas_to_shape_3D(theta, species, radius=0.5, center_mass=1.0, patch_mass=1e-5):
# def thetas_to_shape(thetas_and_phis, species, radius=0.5, center_mass=1.0, patch_mass=1e-5):
  # Convert angles to patch positions
  thetas_and_phis = jnp.array([0, jnp.pi*(theta/180), jnp.pi*(theta/180), jnp.pi*(theta/180),
                             0, 0, jnp.pi*2/3, -jnp.pi*2/3])
  thetas = thetas_and_phis[:NUM_PATCHES]
  phis = thetas_and_phis[NUM_PATCHES:]
  patch_positions = jnp.zeros((NUM_PATCHES, 3), dtype = jnp.float64)
  patch_positions = patch_positions.at[:, 0].set(radius*jnp.cos(phis)*jnp.sin(thetas))
  patch_positions = patch_positions.at[:, 1].set(radius*jnp.sin(phis)*jnp.sin(thetas)) 
  patch_positions = patch_positions.at[:, 2].set(radius*jnp.cos(thetas))

  # Add the central particle (position (0, 0)) to the list of positions
  central_particle_position = jnp.array([[0.0, 0.0, 0.0]])
  positions = jnp.concatenate((central_particle_position, patch_positions), axis=0)

  if not len(thetas)+1 == len(species):
    raise ValueError("Must provide species information for every particle")

  # Combine positions and species to form a patchy particle rigid_body object
  masses = jnp.array([center_mass] + len(thetas)*[patch_mass])
  shape = rigid_body.point_union_shape(positions, masses).set(point_species=species)
  return shape

### body_to_plot converts rigid_body definitions to a sequence of positions
### can be used for plotting
### there is no dramatic difference between 2D and 3D except the reshape function
@jit
def body_to_plot_2D(body, thetas, species, radius=0.5):
  shape = thetas_to_shape(thetas, species, radius=radius)
  body_pos = vmap(rigid_body.transform, (0, None))(body, shape)
  bodypos = body_pos.reshape(-1, 2)
  species_list = jnp.array(list(species) * len(body.center))

  inds_at_id = lambda id: jnp.squeeze(jnp.argwhere(species_list==id))
  center_particles = bodypos[inds_at_id(0)]
  list_of_patch_particles = []
  for i in range(1, len(species)):
    list_of_patch_particles += [bodypos[inds_at_id(i)]]

  return center_particles, list_of_patch_particles

@jit
def body_to_plot(body, thetas_and_phis, species, radius=0.5):
  shape = thetas_to_shape(thetas_and_phis, species, radius=radius)
  body_pos = vmap(rigid_body.transform, (0, None))(body, shape)
  bodypos = body_pos.reshape(-1, 3)
  species_list = jnp.array(list(species) * len(body.center))

  inds_at_id = lambda id: jnp.squeeze(jnp.argwhere(species_list==id))
  center_particles = bodypos[inds_at_id(0)]
  list_of_patch_particles = []
  for i in range(1, len(species)):
    list_of_patch_particles += [bodypos[inds_at_id(i)]]

  return center_particles, list_of_patch_particles

### plot_traj shows a nice example to plot 2D patchy particle trajectories using
### matplotlib's EllipseCollection function, using this function, you can draw
### accurate geometries of the patchy particles
### For 3D visualization, converting the trajectory into one of the standard MD trajectory
### format (.xyz, .gsd, etc) and use OVITO is probably the best choice
def plot_traj(trajectory, thetas, species, colors, box_size, radius=0.5, p_radius=0.14):
  center_pos, patches_pos = body_to_plot(trajectory, thetas, species, radius=radius)
  fig, axs = plt.subplots(1)
  axs.add_collection(EllipseCollection(widths=radius*2, heights=radius*2, angles=0, units='xy',
                                       facecolors=colors[0], alpha = 0.6, 
                                       offsets=list(center_pos), transOffset=axs.transData))
  for k in range(1, NUM_SPECIES):
    axs[i, j].add_collection(EllipseCollection(widths=p_radius*2, heights=p_radius*2, angles=0, units='xy',
                             facecolors=colors[k], edgecolors=colors[k], alpha = 1.0, 
                             offsets=list(patches_pos[k-1]), transOffset=axs.transData))

    axs.set_xlim([0, box_size])
    axs.set_ylim([0, box_size])
    axs.set_axis_off()

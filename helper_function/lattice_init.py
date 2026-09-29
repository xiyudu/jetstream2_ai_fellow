### index_list: number of unitcells in each dimension
### a0/a1: primitive vectors
### unitcell: unitcell definition for lattice/crystal
### offset: noise from perfect lattice/crystal
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

def make_3D_lattice_from_index_list(index_list,
                                    a0=jnp.array([jnp.sqrt(2),0.0,0.0]),
                                    a1=jnp.array([0.0,jnp.sqrt(2),0.0]),
                                    a2=jnp.array([0.0,0.0,jnp.sqrt(2)]),
                                    unitcell=jnp.array([[0.0, 0.0, 0.0]]),
                                    offset=jnp.array([0.0,0.0,0.0])):
  unitcell_particles = unitcell + offset
  N = len(unitcell_particles)*index_list.shape[0]
  def get_cell(i,j,k):
    cell_shift = i*a0+j*a1+k*a2
    return unitcell_particles + cell_shift

  return jnp.reshape(vmap(get_cell, in_axes=(0,0,0))(index_list[:,0],index_list[:,1],index_list[:,2]),(N,3))

def make_3D_lattice(irange, jrange, krange, **kwargs):
  index_list = jnp.array([ [i,j,k] for i in irange for j in jrange for k in krange])
  return make_3D_lattice_from_index_list(index_list, **kwargs)

make_3D_lattice = jit(make_3D_lattice)

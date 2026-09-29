### This function computes a differentiable-version of the
### k-atic order parameter
### displacement_all: functions to compute the distance between
### all particles
### k: k-fold symmetry to target
### r0: nearest neighbor distance cut_off (to ensure differentiability,
###     we use a smooth function to weight particles within the first
###     neighbor shell, you can also change it for larger values to include the
###     second/third neighbor shells)
### alpha: steepness of the smoothing function
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

### This function computes a differentiable-version of the
### local bond order parameter (Steinhardt Order Parameter)
### displacement_all: functions to compute the distance between
### all particles
### l: l-fold symmetry to target for spherical harmonics
### r0: nearest neighbor distance cut_off (to ensure differentiability,
###     we use a smooth function to weight particles within the first
###     neighbor shell, you can also change it for larger values to include the
###     second/third neighbor shells)
### alpha: steepness of the smoothing function
def get_Q_l(displacement_all, l = 6, r0 = 1.1, alpha = 100):

  def Ylm(theta, phi):
    m = jnp.arange(-l, l + 1)
    return scipy.special.sph_harm_y(jnp.array([l]), m, 
                                    jnp.array([theta]), jnp.array([phi]))

  def get_ylms(dR_ij):
    epsilon = 0.00001 #avoids nan in derivative
    dR_ij = jnp.where(dR_ij==0.0, epsilon, dR_ij)
    phi = jnp.arctan2(dR_ij[1], dR_ij[0])
    rsin = jnp.sqrt(dR_ij[0]**2 + dR_ij[1]**2)
    theta = jnp.arctan2(rsin, dR_ij[2])
    return Ylm(theta, phi)

  def weight(r):
    return jnp.where(r<1e-7, 0., 1.0/(1 + jnp.exp(alpha*(r - r0))))

  def calculate_order_param(R):
    v_get_ylms = vmap(vmap(get_ylms))
    ds = displacement_all(R, R)
    r = space.distance(ds)
    w = jnp.expand_dims(weight(r), -1)
    q_lm = (jnp.sum(v_get_ylms(ds)*w, axis=0)/(jnp.sum(w, axis=0)+0.000001)).reshape(-1,1,l*2+1)
    q_lm_broad = jnp.repeat(q_lm, repeats=len(q_lm), axis=1)
    ave_q_lm = jnp.sum(q_lm_broad*w, axis=0)/(jnp.sum(w, axis=0)+0.000001)
    Qlm_each_particle = jnp.sqrt(4*jnp.pi*jnp.abs(jnp.einsum('...m, ...m', jnp.conj(ave_q_lm), ave_q_lm))/(l*2+1))
    return jnp.mean(Qlm_each_particle)

  return calculate_order_param

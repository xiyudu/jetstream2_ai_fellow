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

  def weight(r, r0=1.1, alpha=100):
    return jnp.where(r<1e-7, 0., 1.0/(1 + np.exp(alpha*(r - r0))))

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
    w = weight(r, r0=r0, alpha=alpha)
    psi_k = (np.sum(v_get_ylms(ds)*w, axis=0)/(jnp.sum(w, axis=0)+0.00001))
    return psi_k

  def average_order_param(R):
    psi_k = calculate_order_param(R)
    return jnp.abs(jnp.mean(psi_k))

  return average_order_param

### 1. Setup
Follow the Jetstream2 [documentation](https://docs.jetstream-cloud.org/getting-started/first-instance/) to set up an instance with the latest Ubuntu Linux distribution (this process might take a couple minutes). Make sure to select a GPU instance so that we can then properly install the GPU version of `jax` and `jax-md`. 

After the instance is set up, launch the instance.

We will use `pip3` to install `jax` and `jax-md` and we want to do it locally. To properly install the desired packages, we will need to make a local `python3` environment using the following command:
```
python3 -m venv local/jax_md_install
```
Here, I created a folder called `jax_md_install` under the `local` folder I created to store potential software packages.

After local environment setup, we will put the following lines in our `.bashrc` file so that we use the correct `python` distribution.
```
alias python="/home/exouser/local/jax_md_install/bin/python3"
alias pip="/home/exouser/local/jax_md_install/bin/pip3"
```
Then we will follow the installation instructions for [`jax`](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) and [`jax-md`](https://github.com/jax-md/jax-md?tab=readme-ov-file) on their GitHub Page to install the two packages.
```
pip install -U "jax[cuda13]"
pip install jax-md --upgrade
```
(Note: the desired CUDA version might change for `jax` so always refer back to the GitHub link.)

After everything is installed, here is a sample $NVE$ simulation to test the installation.
<details>

<summary>Sample Script</summary>

```python
import numpy as onp
from jax import config
config.update('jax_enable_x64', True)

import jax.numpy as np
from jax import random
from jax import jit
from jax import lax

import time
import os

from jax_md import space, smap, energy, minimize, quantity, simulate

def square_lattice(N, box_size):

  Nx = int(np.sqrt(N))
  Ny, ragged = divmod(N, Nx)
  if Ny != Nx or ragged:
    assert ValueError('Particle count should be a square. Found {}.'.format(N))
  length_scale = box_size / Nx

  R = []
  for i in range(Nx):
    for j in range(Ny):
      R.append([i * length_scale, j * length_scale])

  return np.array(R)

N = 400
ms = 10
dimension = 2
rho = 0.05
box_size = (N/rho)**(1/2)

displacement, shift = space.periodic(box_size)

key = random.PRNGKey(0)
R = square_lattice(N, N**(1/2)*2.0)
  
energy_fn = energy.lennard_jones_pair(displacement)
E = energy_fn(R)
print("Energy of the system is: ", E)

init, apply = simulate.nve(energy_fn, shift, 1e-2)
step = jit(lambda i, state: apply(state))
state = init(key, R, kT=0.0)

PE = [energy_fn(state.position)]
KE = [quantity.kinetic_energy(momentum=state.momentum)]
P = [state.momentum]
trajectory = [state.position]

N_steps = 2000
inner_steps = 10
print_every = 100
old_time = time.time()

print('Step\tKE\tPE\tTotal Energy\ttime/step')
print('----------------------------------------')

for i in range(N_steps):

  state = lax.fori_loop(0, inner_steps, step, state)  
  PE += [energy_fn(state.position)]
  KE += [quantity.kinetic_energy(momentum=state.momentum)]
  P += [state.momentum]
  trajectory += [state.position]

  if i % print_every == 0 and i > 0:
    new_time = time.time()
    print(
      '{}\t{:.2f}\t{:.2f}\t{:.3f}\t{:.2f}'.format(
        i * inner_steps,
        KE[-1],
        PE[-1],
        KE[-1] + PE[-1],
        (new_time - old_time) / print_every / inner_steps,
        )
      )
    old_time = new_time

PE = np.array(PE)
KE = np.array(KE)
R = state.position
```
</details>

### 2. Re-Initialization
Jetstream2 allows us to shelve an instance or create an image so that our setup is not lost. To do that, follow the instructions in the [documentation](https://docs.jetstream-cloud.org/getting-started/snapshots/)

### 3. Connect to Jupyter Notebook
Since we had to set up our own `Python` environment to install the relevant packages, we cannot use the default `jupyter-ip.sh` script to launch a remote Jupyter notebook. We need to set up remote access using the following commands:
```
jupyter-lab --no-browser --port=8080 #this port number can be any number, just need to be consistent throughout
ssh -L 8080:localhost:8080 exouser@<REMOTE_HOST>
```
The remote `Jupyter Notebook` can be accessed at
```

### 4. Running when Log Out
```
nohup /home/exouser/local/jax_md_install/bin/python3 /home/exouser/project/opt_test/{code}
```
http://localhost:8080/
```

# The following code snippet will be run on all TPU hosts
import jax
import jax.numpy as jnp
from jax import random

jax.distributed.initialize()

print('== Running worker: ', jax.process_index())

def dot(v1, v2):
  return jnp.vdot(v1, v2)

rng_key = random.PRNGKey(42 + 10*jax.process_index())

vs = random.normal(rng_key, shape=(2_000_000,3))
v1s = vs[:1_000_000,:]
v2s = vs[1_000_000:,:]

# The total number of TPU cores in the Pod
device_count = jax.device_count()

# The number of TPU cores attached to this host
local_device_count = jax.local_device_count()

if jax.process_index() == 0:
    print('-- global device count:', jax.device_count())
    #print('global devices:', jax.devices())
    print('-- local device count:', jax.local_device_count())
    #print('local devices:', jax.local_devices())
    print('-- JAX version:', jax.__version__)

v1sp = v1s.reshape((local_device_count, v1s.shape[0]//local_device_count, v1s.shape[1]))
v2sp = v2s.reshape((local_device_count, v2s.shape[0]//local_device_count, v2s.shape[1])) 

if jax.process_index() == 0:
    print('-- v1sp shape: ', v1sp.shape)

dots = jax.pmap(jax.vmap(dot))(v1sp,v2sp)    # (8, 125K)
if jax.process_index() == 0:
    print('-- dots shape: ', dots.shape)

global_sum = jax.pmap(
    lambda x: jax.lax.psum(jnp.sum(x), axis_name='p'),
    axis_name='p'
)(dots)

if jax.process_index() == 0:
    print('-- global_sum shape: ', global_sum.shape)
print(f'== Worker {jax.process_index()} global sum: {global_sum}')

dots = dots.reshape((dots.shape[0]*dots.shape[1]))
if jax.process_index() == 0:
    print('-- result shape: ', dots.shape)
local_sum = jnp.sum(dots)
print(f'== Worker {jax.process_index()} local sum: {local_sum}')

print(f'== Worker {jax.process_index()} done')

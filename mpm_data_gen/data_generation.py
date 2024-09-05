import taichi as ti
import math
import random
import numpy as np
import argparse

ti.init(arch=ti.gpu)

quality = 1
n_particles, n_grid = 5000 * quality**2, 128 * quality
print(n_particles)
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho
E, nu = 1e3, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
material = ti.field(dtype=int, shape=n_particles)  # material id
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass
gravity = ti.Vector.field(2, dtype=float, shape=())
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(2, dtype=float, shape=())
@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]

        h = .3
        mu, la = mu_0 * h, lambda_0 * h
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig

        F[p] = U @ sig @ V.transpose()

        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 2) * la * J * (
            J - 1
        )
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]
            grid_v[i, j] += dt * gravity[None] * 30
            dist = attractor_pos[None] - dx * ti.Vector([i, j])
            grid_v[i, j] += dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0
            if i > n_grid - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]

def circle_point(center, area):
    max_r = ti.sqrt(area / math.pi)
    r = ti.sqrt(
        ti.random()) * max_r  # Radius scaled by square root of random value to ensure uniform distribution
    theta = ti.random() * 2 * math.pi  # Random angle in radians

    x = [
        r * ti.cos(theta) + center[0],  # X coordinate
        r * ti.sin(theta) + center[1]  # Y coordinate
    ]
    return x

def triangle_point(center, area):
    side_length = ti.sqrt((4 * area) / math.sqrt(3))
    height = (math.sqrt(3) / 2) * side_length

    A = ti.Vector([center[0] - side_length / 2, center[1] - height / 3])
    B = ti.Vector([center[0] + side_length / 2, center[1] - height / 3])
    C = ti.Vector([center[0], center[1] + 2 * height / 3])

    u = ti.random()
    v_barycentric = ti.random()

    sqrt_u = ti.sqrt(u)
    u = 1 - sqrt_u
    v_barycentric = v_barycentric * sqrt_u

    w = 1 - u - v_barycentric
    x = u * A + v_barycentric * B + w * C

    return x

def square_point(center, area):
    side_length = ti.sqrt(area)
    x = [
        center[0] + (ti.random() - 0.5) * side_length,
        center[1] + (ti.random() - 0.5) * side_length
    ]

    return x

shape_one = random.randint(1, 3)
shape_two = random.randint(1, 3)
shape_three = random.randint(1, 3)
s1, s2, s3 = random.randint(5,20), random.randint(5,20), random.randint(5,20)
d1, d2, d3 = random.randint(0,360), random.randint(0,360), random.randint(0,360)
c1 = [random.random() * .8 + .1, random.random() * .3 + .6]
c2 = [random.random() * .3 + .1, random.random() * .3 + .1]
c3 = [random.random() * .3 + .6, random.random() * .3 + .1]
a1, a2, a3 = random.random() * .012 + .008, random.random() * .012 + .008, random.random() * .012 + .008

@ti.kernel
def reset():
    for i in range(n_particles):
        if i < n_particles * .33:
            center = c1
            area = a1

            if shape_one == 1:
                x[i] = circle_point(center, area)
            elif shape_one == 2:
                x[i] = square_point(center, area)
            else:
                x[i] = triangle_point(center, area)

            angle_rad = math.radians(d1)

            vx = s1 * ti.cos(angle_rad)  # X component of velocity
            vy = s1 * ti.sin(angle_rad)

            v[i] = [vx, vy]

        elif i < n_particles * .66:
            center = c2
            area = a2

            if shape_two == 1:
                x[i] = circle_point(center, area)
            elif shape_two == 2:
                x[i] = square_point(center, area)
            else:
                x[i] = triangle_point(center, area)

            angle_rad = math.radians(d2)

            vx = s2 * ti.cos(angle_rad)  # X component of velocity
            vy = s2 * ti.sin(angle_rad)

            v[i] = [vx, vy]

        else:
            center = c3
            area = a3

            if shape_three == 1:
                x[i] = circle_point(center, area)
            elif shape_three == 2:
                x[i] = square_point(center, area)
            else:
                x[i] = triangle_point(center, area)

            angle_rad = math.radians(d3)

            vx = s3 * ti.cos(angle_rad)  # X component of velocity
            vy = s3 * ti.sin(angle_rad)

            v[i] = [vx, vy]

        material[i] = 1

        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, 2, 2)


reset()
time_steps = []

gravity[None] = [0, -1]
counter = 0
for frame in range(300):
    for s in range(int(2e-3 // dt)):
        counter += 1
        time_steps.append(x.to_numpy().copy())
        substep()

final_array = np.stack(time_steps, axis=0)
parser = argparse.ArgumentParser(description='Save NumPy array with run number')
parser.add_argument('--run_number', type=int, required=True, help='Run number for the file')

args = parser.parse_args()

# Define the file path where you want to save the array
file_path = '/Users/mrohatgi/MPM_TRAINING/run_' + str(args.run_number)
np.savez(file_path, data=final_array)
# The flow and reaction was solved by Lattice Boltzmann Method using Taichi language
# Auther: Tau (Lizt1191@gmail.com)
# Time: February 2023

import taichi as ti
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import time

ti.init(arch=ti.gpu)


@ti.data_oriented
class Reaction_Flow:
    def __init__(self, Re):
        nx = 320
        ny = 320
        self.nx = nx
        self.ny = ny
        self.rho0 = 1.0
        self.C_s0 = 1.0
        self.Lx = 320.0
        self.Ly = 320.0
        self.umax = 0.06
        self.K_s = 1.0
        self.Re = Re
        self.niu = self.umax * self.Ly / self.Re
        self.C = 1.0
        self.Cs = self.C / np.sqrt(3)
        self.dx = self.Ly / self.ny
        self.dt = self.dx / self.C
        self.Rc = 1 / self.C
        self.Rcc = self.Rc ** 2
        self.C2 = self.C ** 2
        self.Cs2 = self.Cs ** 2
        self.tau = self.niu / (self.Cs2 * self.dt) + 0.5
        self.w = 1.0 / self.tau
        self.Ds = 1.0 / 6
        self.tau_s = self.Ds / (self.Cs2 * self.dt) + 0.5
        self.w_s = 1.0 / self.tau_s
        self.ax = 8 * self.umax * self.niu / self.Ly / self.Ly
        self.Vel_Temp = self.C_s_Temp = None
        self.Start_Time = self.End_Time = None

        self.rho = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.C_s = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
        self.fg = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.f = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.gg = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.g = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.omega_i = ti.field(dtype=ti.f32, shape=9)
        self.e = ti.field(dtype=ti.i32, shape=(9, 2))

        arr = np.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0,
                        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0], dtype=np.float32)
        self.omega_i.from_numpy(arr)
        arr = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1],
                        [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)
        self.e.from_numpy(arr)

    @ti.func
    def f_eq(self, i, j, k):
        eu = (ti.f32(self.e[k, 0]) * self.vel[i, j][0]
              + ti.f32(self.e[k, 1]) * self.vel[i, j][1]) * self.Rc
        uv = (self.vel[i, j][0] ** 2.0 + self.vel[i, j][1] ** 2.0) * self.Rcc
        return self.omega_i[k] * self.rho[i, j] * \
            (1.0 + 3.0 * eu + 4.5 * eu ** 2 - 1.5 * uv)

    @ti.func
    def g_eq(self, i, j, k):
        eu = (ti.f32(self.e[k, 0]) * self.vel[i, j][0]
              + ti.f32(self.e[k, 1]) * self.vel[i, j][1]) * self.Rc
        uv = (self.vel[i, j][0] ** 2.0 + self.vel[i, j][1] ** 2.0) * self.Rcc
        return self.omega_i[k] * self.C_s[i, j] * \
            (1.0 + 3.0 * eu + 4.5 * eu ** 2 - 1.5 * uv)

    @ti.func
    def force(self, i, j, k):
        f1 = self.Rc * ti.f32(self.e[k, 0])
        f2 = ((ti.f32(self.e[k, 0]) ** 2) * self.vel[i, j][0] + ti.f32(self.e[k, 1]) * ti.f32(self.e[k, 1]) *
              self.vel[i, j][1]) * self.Rcc
        f3 = self.vel[i, j][0] * self.Rcc
        return self.omega_i[k] * self.rho[i, j] * self.ax * (1 - 0.5 * self.w) * (3 * f1 + 9 * f2 - 3 * f3)

    @ti.kernel
    def init(self):
        for i, j in self.rho:
            self.vel[i, j][0] = 0.0
            self.vel[i, j][1] = 0.0
            self.rho[i, j] = self.rho0
            self.C_s[i, j] = 0.0
            self.C_s[0, j] = self.C_s0
            for k in ti.static(range(9)):
                self.f[i, j][k] = self.f_eq(i, j, k)
                self.g[i, j][k] = self.g_eq(i, j, k)

    @ti.kernel
    def collide(self):
        for i, j, k in ti.ndrange(self.nx, self.ny, 9):
            self.fg[i, j][k] = self.f[i, j][k] - (self.f[i, j][k] - self.f_eq(i, j, k)) * self.w \
                               + self.dt * self.force(i, j, k)
            self.gg[i, j][k] = self.g[i, j][k] - (self.g[i, j][k] - self.g_eq(i, j, k)) * self.w_s

    @ti.kernel
    def stream(self):
        for i, j, k in ti.ndrange((0, self.nx), (1, self.ny - 1), 9):
            id = (i - self.e[k, 0] + self.nx) % self.nx
            jd = j - self.e[k, 1]
            self.f[i, j][k] = self.fg[id, jd][k]
            if i != 0 and i != self.nx - 1:
                self.g[i, j][k] = self.gg[id, jd][k]

    @ti.kernel
    def update_macro_var(self):
        for i, j in ti.ndrange((0, self.nx), (1, self.ny - 1)):
            self.rho[i, j] = self.vel[i, j][0] = self.vel[i, j][1] = self.C_s[i, j] = 0.0
            for k in ti.static(range(9)):
                self.rho[i, j] += self.f[i, j][k]
                self.vel[i, j][0] += self.C * (ti.f32(self.e[k, 0]) *
                                               self.f[i, j][k])
                self.vel[i, j][1] += self.C * (ti.f32(self.e[k, 1]) *
                                               self.f[i, j][k])
                if i != 0 and i != self.nx - 1:
                    self.C_s[i, j] += self.g[i, j][k]
            self.vel[i, j][0] /= self.rho[i, j]
            self.vel[i, j][1] /= self.rho[i, j]
            self.vel[i, j][0] += 0.5 * self.ax * self.dt

    @ti.kernel
    def wall_bc(self):
        for i in ti.ndrange((0, self.nx)):
            self.vel[i, self.ny - 1][0] = 0.0
            self.vel[i, self.ny - 1][1] = 0.0
            self.rho[i, self.ny - 1] = self.rho[i, self.ny - 2]
            self.C_s[i, self.ny - 1] = self.C_s[i, self.ny - 2]
            self.vel[i, 0][0] = 0.0
            self.vel[i, 0][1] = 0.0
            self.rho[i, 0] = self.rho[i, 1]
            self.C_s[i, 0] = self.C_s[i, 1] - 0.5 * self.dx * self.K_s * self.C_s[i, 1] / self.Ds
            for k in ti.static(range(9)):
                self.f[i, self.ny - 1][k] = self.f_eq(i, self.ny - 1, k) + (
                        self.f[i, self.ny - 2][k] - self.f_eq(i, self.ny - 2, k))
                self.f[i, 0][k] = self.f_eq(i, 0, k) + (self.f[i, 1][k] - self.f_eq(i, 1, k))
                self.g[i, self.ny - 1][k] = self.g_eq(i, self.ny - 1, k) + (
                        self.g[i, self.ny - 2][k] - self.g_eq(i, self.ny - 2, k))
                self.g[i, 0][k] = self.g_eq(i, 0, k) + (self.g[i, 1][k] - self.g_eq(i, 1, k))
        for j in ti.ndrange((1, self.ny - 1)):
            self.vel[self.nx - 1, j][0] = self.vel[self.nx - 1, j][0]
            self.vel[self.nx - 1, j][1] = self.vel[self.nx - 1, j][1]
            self.C_s[self.nx - 1, j] = self.C_s[self.nx - 2, j]
            self.vel[0, j][0] = self.vel[0, j][0]
            self.vel[0, j][1] = 0.0
            self.C_s[0, j] = self.C_s0
            for k in ti.static(range(9)):
                self.g[0, j][k] = self.g_eq(0, j, k) + (self.g[1, j][k] - self.g_eq(1, j, k))
                self.g[self.nx - 1, j][k] = self.g_eq(self.nx - 1, j, k) + (
                        self.g[self.nx - 2, j][k] - self.g_eq(self.nx - 2, j, k))

    def solve(self):
        gui = ti.GUI('Reaction_Flow', (self.nx, 2 * self.ny))
        self.Start_Time = time.time()
        i = 0
        self.init()
        while 1:
            i = i + 1
            if i % 1000 == 0:
                self.vel_Temp = self.vel.to_numpy()
                self.C_s_Temp = self.C_s.to_numpy()
            self.collide()
            self.stream()
            self.update_macro_var()
            self.wall_bc()
            if i % 100 == 0:
                vel = self.vel.to_numpy()
                dens = self.C_s.to_numpy()
                vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5
                dens_mag = dens[:, :]
                vel_img = cm.plasma(vel_mag / 0.015)
                dens_img = cm.plasma(dens_mag)
                img = np.concatenate((dens_img, vel_img), axis=1)
                gui.set_image(img)
                gui.show()
            if i % 1000 == 0:
                self.End_Time = time.time()
                Time = int(self.End_Time - self.Start_Time)
                All_Time = Time
                hou = Time / 3600
                Time = Time % 3600
                mint = Time / 60
                Time = Time % 60
                sec = Time
                U_Err_Temp = ((self.vel.to_numpy()[:, :, 0] - self.vel_Temp[:, :, 0]) ** 2 + (
                        self.vel.to_numpy()[:, :, 1] - self.vel_Temp[:, :, 1]) ** 2) ** 0.5
                U_Temp2 = (self.vel.to_numpy()[:, :, 0] ** 2 + self.vel.to_numpy()[:, :, 1] ** 2) ** 0.5
                C_s_Err_Temp = self.C_s.to_numpy() - self.C_s_Temp
                C_s_Temp2 = self.C_s.to_numpy()
                U_Err = U_Err_Temp.sum() / U_Temp2.sum()
                C_s_Err = C_s_Err_Temp.sum() / C_s_Temp2.sum()
                print("U_center= %e,C_center= %e,U_Err= %e,C_Err %e,Step: %d,Time: %02d %02d:%02d:%02d" % (
                    self.vel[160, 160][0],
                    self.C_s[160, 160],
                    U_Err,
                    C_s_Err,
                    i,
                    All_Time / i * 1000,
                    hou, mint, sec))
                if U_Err <= 1.0e-6 and C_s_Err < 1.0e-6:
                    global C_s_End
                    C_s_End = self.C_s.to_numpy()[:, :]
                    fig, Comp2 = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=200)
                    x_lbm = np.linspace(10, 320.0, 31)
                    x_analyse = np.linspace(1, 320, 320)
                    Comp2.plot(x_lbm, (C_s_End[10:320:10, 1] - C_s_End[10:320:10, 0]) * 320,
                               'b^',
                               label='LBM Solution')
                    Comp2.plot(x_analyse, 0.854 * (0.06 * 320 * 320 / x_analyse * 6) ** (1 / 3),
                               'r-',
                               label='Analysis Solution')
                    Comp2.set_xlabel(r'Y/Ly')
                    Comp2.set_ylabel(r'U')
                    Comp2.legend()
                    plt.tight_layout()
                    plt.show()
                    break

if __name__ == '__main__':
    Reaction_Flow(20).solve()





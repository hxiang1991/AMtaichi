import os
import math
import time
import numpy as np
import taichi as ti
import taichi.math as tm
from arguments import args
from scipy.interpolate import interpn
from utils import laser_track, get_unique_ori, post_process

import matplotlib.cm as cm

ti.init(arch=ti.gpu, default_ip=ti.i32, default_fp=ti.f32, device_memory_fraction=0.95, random_seed=int(time.time()))
vec2_f = ti.types.vector(2, ti.f32)
vec3_i = ti.types.vector(3, ti.i32)
vec3_f = ti.types.vector(3, ti.f32)

ti.sync()
ti.init(kernel_profiler=True)


@ti.data_oriented
class PowderDEM:

    def __init__(self):

        self.Nx, self.Ny, self.dxyz = args.Nx, args.Ny, args.dxyz
        self.dxyzi = 1 / self.dxyz
        self.Nz = args.Nz_old + args.layer_Nz * (1 + (args.Nz_old == 0))

        self.num_powder = args.num_powder + args.num_powder * (args.Nz_old == 0)
        self.powder_r_min, self.powder_r_max = args.powder_r_min, args.powder_r_max
        self.r_dmin = ti.floor(self.powder_r_min / self.dxyz)
        self.r_dmax = ti.floor(self.powder_r_max / self.dxyz)

        self.interface_thick = 5  # grain boundary thickness
        self.t_mesh = self.interface_thick // 2  # GB thickness with phi value from 0 to 0.5

        args.Nz = args.Nz_old + args.layer_Nz * (1 + (args.Nz_old == 0))
        args.domain_height = args.Nz / (1 / args.dxyz)
        args.total_height = args.layer_thick * (args.num_layer + 1)
        args.total_Nz = int(args.total_height / self.dxyz)

        # to generate the grain distribution within powder particles by simple phase field method,
        # note that individual metal powder may contain several grains
        self.num_oris = args.num_oris
        self.dt, self.nstep = args.dt_PFM, 1000
        self.mobil, self.mag, self.kappa = 1e00, 1e06, 4e-07
        self.pf = ti.Struct.field({
            'zeta_': float,
            'phi': float,
            'phi_td': float,
            'ori_id': ti.int32,
            'sum_phi2': float,
        }, shape=(self.Nx, self.Ny, args.total_Nz, self.num_oris))

        self.sum_phi2 = ti.field(dtype=float, shape=(args.Nx, args.Ny, args.total_Nz))  ###-------------

        # z height container of upper surface for each point in x-y plane
        if args.Nz_old == 0:
            args.z_height = ti.field(dtype=ti.int32, shape=1)
            args.zeta = ti.field(dtype=float, shape=(args.Nx, args.Ny, args.total_Nz))
            # 0 ~ -1: spatial distribution of grain orientation; -1: orientation id of grains
            args.phi = ti.field(dtype=float, shape=(args.Nx, args.Ny, args.total_Nz,  self.num_oris + 1))

        self.iflag = ti.field(dtype=ti.int32, shape=1)  # for debug

    @ti.kernel
    def pfm_solver(self, step: ti.int32, zeta: ti.template(), phi: ti.template(), pf: ti.template()):

        if step == 0:
            for ix, iy, iz, g in ti.ndrange(self.Nx, self.Ny, self.Nz, self.num_oris):
                pf[ix, iy, iz, g].phi = 2e-03 + 1e-03 * (ti.random(float) - 0.5)

        for ix, iy, iz, g in ti.ndrange(self.Nx, self.Ny, args.total_Nz, self.num_oris):
            pf[ix, iy, iz, g].phi_td = pf[ix, iy, iz, g].phi

        # ------ phi update------------
        x0, x1 = 1, self.Nx - 1
        y0, y1 = 1, self.Ny - 1
        z0, z1 = 1, args.total_Nz - 1
        for ix, iy, iz, g in ti.ndrange((x0, x1), (y0, y1), (z0, z1), self.num_oris):

            phi0 = pf[ix, iy, iz, g].phi_td
            _term = 0.
            for gg in range(self.num_oris):
                _term += phi0 * pf[ix, iy, iz, gg].phi_td ** 2
            _term -= phi0 ** 3
            dfdphi = self.mag * (-phi0 + phi0 ** 3 + 2 * _term)

            a_e, a_w = self.mobil * self.kappa * self.dxyz, self.mobil * self.kappa * self.dxyz
            a_n, a_s = self.mobil * self.kappa * self.dxyz, self.mobil * self.kappa * self.dxyz
            a_f, a_b = self.mobil * self.kappa * self.dxyz, self.mobil * self.kappa * self.dxyz
            ap0 = self.dxyz ** 3 / self.dt
            ap = ap0 + a_e + a_w + a_n + a_s + a_f + a_b
            rhs = (ap0 * pf[ix, iy, iz, g].phi_td - dfdphi * self.dxyz ** 3 +
                   a_e * pf[ix + 1, iy, iz, g].phi_td + a_e * pf[ix - 1, iy, iz, g].phi_td +
                   a_n * pf[ix, iy + 1, iz, g].phi_td + a_s * pf[ix, iy - 1, iz, g].phi_td +
                   a_f * pf[ix, iy, iz + 1, g].phi_td + a_b * pf[ix, iy, iz - 1, g].phi_td)

            pf[ix, iy, iz, g].phi = tm.clamp(rhs / ap, 0., 1.)

        for ix, iy, iz, g in ti.ndrange(self.Nx, self.Ny, args.total_Nz, self.num_oris):

            pf[0, iy, iz, g].phi = pf[1, iy, iz, g].phi
            pf[ix, 0, iz, g].phi = pf[ix, 1, iz, g].phi
            pf[ix, iy, 0, g].phi = pf[ix, iy, 1, g].phi

            pf[self.Nx - 1, iy, iz, g].phi = pf[self.Nx - 2, iy, iz, g].phi
            pf[ix, self.Ny - 1, iz, g].phi = pf[ix, self.Ny - 2, iz, g].phi
            pf[ix, iy, self.Nz - 1, g].phi = pf[ix, iy, self.Nz - 2, g].phi

        if step == self.nstep - 1:
            for ix, iy, iz, g in ti.ndrange(self.Nx, self.Ny, args.total_Nz, self.num_oris):
                self.sum_phi2[ix, iy, iz] += phi[ix, iy, iz, g] ** 2

            for ix, iy, iz, g in ti.ndrange(self.Nx, self.Ny, args.total_Nz, self.num_oris):
                if phi[ix, iy, iz, g] <= 0.5 and self.sum_phi2[ix, iy, iz] <= 0.25:  #####---------------
                    phi[ix, iy, iz, g] = zeta[ix, iy, iz] * pf[ix, iy, iz, g].phi

                if phi[ix, iy, iz, g] >= 0.5:
                    phi[ix, iy, iz, self.num_oris] = g + 1

    @ti.kernel
    def pfm_solver_test(self, zeta: ti.template(), pf: ti.template(), step: ti.int32):

        if step == 0:
            for ix, iy, iz, g in ti.ndrange(self.Nx, self.Ny, self.Nz, self.num_oris):
                pf[ix, iy, iz, g].phi = 2e-03 + 1e-03 * (ti.random(float) - 0.5)

        for i in ti.grouped(pf):
            pf[i].phi_td = pf[i].phi

        # ------ phi update------------
        x0, x1 = 1, self.Nx - 1
        y0, y1 = 1, self.Ny - 1
        z0, z1 = 1, self.Nz - 1
        for ix, iy, iz, g in ti.ndrange((x0, x1), (y0, y1), (z0, z1), self.num_oris):

            phi0 = pf[ix, iy, iz, g].phi_td
            _term = 0.
            for gg in range(self.num_oris):
                _term += phi0 * pf[ix, iy, iz, gg].phi_td ** 2
            _term -= phi0 ** 3
            dfdphi = self.mag * (-phi0 + phi0 ** 3 + 2 * _term)

            a_e, a_w = self.mobil * self.kappa * self.dxyz, self.mobil * self.kappa * self.dxyz
            a_n, a_s = self.mobil * self.kappa * self.dxyz, self.mobil * self.kappa * self.dxyz
            a_f, a_b = self.mobil * self.kappa * self.dxyz, self.mobil * self.kappa * self.dxyz
            ap0 = self.dxyz ** 3 / self.dt
            ap = ap0 + a_e + a_w + a_n + a_s + a_f + a_b
            rhs = (ap0 * pf[ix, iy, iz, g].phi_td - dfdphi * self.dxyz ** 3 +
                   a_e * pf[ix + 1, iy, iz, g].phi_td + a_e * pf[ix - 1, iy, iz, g].phi_td +
                   a_n * pf[ix, iy + 1, iz, g].phi_td + a_s * pf[ix, iy - 1, iz, g].phi_td +
                   a_f * pf[ix, iy, iz + 1, g].phi_td + a_b * pf[ix, iy, iz - 1, g].phi_td)

            pf[ix, iy, iz, g].phi = tm.clamp(rhs / ap, 0., 1.)

        for ix, iy, iz, g in ti.ndrange(self.Nx, self.Ny, self.Nz, self.num_oris):
            pf[0, iy, iz, g].phi = pf[1, iy, iz, g].phi
            pf[ix, 0, iz, g].phi = pf[ix, 1, iz, g].phi
            pf[ix, iy, 0, g].phi = pf[ix, iy, 1, g].phi

            pf[self.Nx - 1, iy, iz, g].phi = pf[self.Nx - 2, iy, iz, g].phi
            pf[ix, self.Ny - 1, iz, g].phi = pf[ix, self.Ny - 2, iz, g].phi
            pf[ix, iy, self.Nz - 1, g].phi = pf[ix, iy, self.Nz - 2, g].phi

        for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):

            pf[ix, iy, iz, 0].sum_phi2 = 0.
            for g in range(self.num_oris):
                pf[ix, iy, iz, 0].sum_phi2 += pf[ix, iy, iz, g].phi ** 2 * zeta[ix, iy, iz]

                if pf[ix, iy, iz, g].phi * zeta[ix, iy, iz] >= 0.5:
                    pf[ix, iy, iz, 0].ori_id = g + 1

    @ti.func
    def zeta_update(self, zeta, pos, r):

        x, y, z = pos
        bins = r + self.t_mesh + 1
        x0, x1 = x - bins, x + bins + 1
        y0, y1 = y - bins, y + bins + 1
        z0, z1 = z - bins, z + bins + 1
        for ix, iy, iz in ti.ndrange((x0, x1), (y0, y1), (z0, z1)):
            tmp_vec = ti.Vector([ix, iy, iz]) - pos
            dist = tm.length(tmp_vec) - r
            tmp = dist * math.pi / self.interface_thick
            zeta0 = (1 - ti.sin(tmp)) / 2
            zeta0 = 0. if tmp > math.pi / 2 else zeta0
            zeta0 = 1. if tmp < -math.pi / 2 else zeta0

            if zeta0 > 0:
                zeta[ix, iy, iz] += zeta0

    @ti.func
    def get_iflag(self, zeta, vec_c, r):

        iflag = 1
        bins = r + self.t_mesh
        x0, x1 = vec_c[0] - bins, vec_c[0] + bins
        y0, y1 = vec_c[1] - bins, vec_c[1] + bins
        z0, z1 = vec_c[2] - bins, vec_c[2] + bins
        for ix, iy, iz in ti.ndrange((x0, x1), (y0, y1), (z0, z1)):
            tmp_vec = ti.Vector([ix, iy, iz]) - vec_c
            dist = tm.length(tmp_vec) - r
            tmp = dist * math.pi / self.interface_thick
            zeta0 = (1 - ti.sin(tmp)) / 2
            zeta0 = 0. if tmp > math.pi / 2 else zeta0
            zeta0 = 1. if tmp < -math.pi / 2 else zeta0

            if zeta0 > 0:
                if args.Nz_old == 0:
                    if zeta[ix, iy, iz] + zeta0 > 1:
                        iflag = 0
                        break
                else:
                    if zeta[ix, iy, iz] + zeta0 > 1.1:
                        iflag = 0
                        break

        return iflag

    def grain_assign(self, zeta, phi, pf):

        for istep in range(self.nstep):
            self.pfm_solver(istep, zeta, phi, pf)

    @ti.kernel
    def single_powder_deposition(self, zeta: ti.template()):

        tmp_radius = (self.r_dmin +
                      ti.floor(ti.random(float) * (self.powder_r_max - self.powder_r_min) / self.dxyz, int))
        r = tm.clamp(tmp_radius, self.r_dmin, self.r_dmax)
        bins = r + self.t_mesh
        x0, x1 = bins + 1, self.Nx - 1 - bins
        y0, y1 = bins + 1, self.Ny - 1 - bins
        z0, z1 = args.z_height[0] + bins + 1, self.Nz - 1 - bins
        ti.loop_config(serialize=True)
        for iz, iy, ix in ti.ndrange((z0, z1), (y0, y1), (x0, x1)):

            c1 = zeta[ix, iy, iz] < 0.5
            c2 = zeta[ix - r, iy, iz] < 0.5
            c3 = zeta[ix + r, iy, iz] < 0.5
            c4 = zeta[ix, iy - r, iz] < 0.5
            c5 = zeta[ix, iy + r, iz] < 0.5
            c6 = zeta[ix, iy, iz - r] < 0.5
            c7 = zeta[ix, iy, iz + r] < 0.5
            if c1 and c2 and c3 and c4 and c5 and c6 and c7:
                vec_c = ti.Vector([ix, iy, iz])
                iflag = self.get_iflag(zeta, vec_c, r)
                if iflag == 1:
                    self.zeta_update(zeta, vec_c, r)
                    break

        for i in ti.grouped(zeta):
            zeta[i] = tm.clamp(zeta[i], 0, 1)

    def powder_deposition(self, zeta, phi, pf):

        for p in range(self.num_powder):
            self.single_powder_deposition(zeta)

        self.grain_assign(zeta, phi, pf)

    def powder_dem(self):

        self.powder_deposition(args.zeta, args.phi, self.pf)


@ti.data_oriented
class CfdFunc:

    def __init__(self):

        self.T_solid, self.T_liquid, self.T_vapor = args.T_solid, args.T_liquid, args.T_vapor
        self.T_ambient, self.T_subplate = args.T_ambient, args.T_subplate
        self.rho_metal, self.rho_gas = args.rho_metal, args.rho_gas
        self.cp_solid, self.cp_liquid, self.cp_gas = args.cp_solid, args.cp_liquid, args.cp_gas
        self.kappa_solid, self.kappa_liquid, self.kappa_gas = args.kappa_solid, args.kappa_liquid, args.kappa_gas
        self.visco_solid, self.visco_liquid, self.visco_gas = args.visco_solid, args.visco_liquid, args.visco_gas

        self.avg_rho_cp = (self.rho_metal * self.cp_solid + self.rho_gas * self.cp_gas) / 2

        self.mo, self.beta, self.sigma = args.mo, args.beta, args.sigma
        self.K_C, self.C_K, self.dsigma_dT = args.K_C, args.C_K, args.dsigma_dT

        self.p0, self.Kb, self.latent_v = args.p0, args.Kb, args.latent_v
        self.gas_const, self.gra_const, self.SB_const = args.gas_const, args.gra_const, args.SB_const

        self.top_surface, self.layer_order = args.domain_height, args.layer_order
        power, self.r_beam = args.laser_power, args.r_beam
        power_fraction, self.layer_thick = args.power_fraction, args.layer_thick
        self.h_conv, self.emissivity = args.h_conv, args.emissivity
        self.laser_coeff = 2 * power * power_fraction / (math.pi * self.r_beam ** 2 * self.layer_thick)

        self.subplate_Nz = 2  # args.layer_Nz
        self.dt, self.dxyz, self.dxyzi = args.dt_CFD, args.dxyz, 1 / args.dxyz
        self.Nx, self.Ny, self.Nz = args.Nx, args.Ny, args.reduced_Nz  # reduced area size in z direction

        args.temp = ti.field(dtype=float, shape=(args.Nx, args.Ny, args.total_Nz))
        self.cfd = ti.Struct.field({
            'rho': float,  # density
            'p': float,  # pressure
            'vel': vec3_f,  # velocity
            'temp': float,  # temperature
            'visco': float,  # viscosity
            'alpha_1': float,  # volume fraction of metal phase

            'gamma': float,  # step function dependent on temperature
            'f_liquid': float,  # liquid fraction of metal phase
            'cp': float,  # equivalent specific heat capacity
            'Kc': float,  # curvature of metal/gas interface
            'kappa': float,  # thermal conductivity
            'rho_cp': float,

            'F_total': vec3_f,

            'q_laser': float,
            'q_con_rad_vap': float,

            'vof_lhs_2': vec3_f,
            'vof_lhs_2_div': float,

            # auxiliary fields for the calculation
            'grad_p': vec3_f,  # gradient of pressure
            'grad_temp': vec3_f,  # gradient of temperature
            'grad_alpha_1': vec3_f,  # gradient of volume fraction of the metal phase
            'norm_grad_alpha_1': vec3_f,
            'leng_grad_alpha_1': float,

            'pt': float,
            'temp_td': float,
            'vel_star': vec3_f,
            'alpha_1_td': float,
            'vel_star_div': float,
            'possion_p_rhs': float,
            }, shape=(self.Nx, self.Ny, self.Nz))

        self.layer_switch = ti.field(dtype=ti.i32, shape=1)
        self.layer_switch[0] = args.layer_switch
        self.iflag = ti.field(dtype=ti.int32, shape=1)  # for debug

    @ti.func
    def grad_var(self, var, var_grad):

        for ix, iy, iz in ti.ndrange((1, self.Nx - 1), (1, self.Ny - 1), (1, self.Nz - 1)):
            var_grad[ix, iy, iz][0] = (var[ix + 1, iy, iz] - var[ix - 1, iy, iz]) * self.dxyzi / 2
            var_grad[ix, iy, iz][1] = (var[ix, iy + 1, iz] - var[ix, iy - 1, iz]) * self.dxyzi / 2
            var_grad[ix, iy, iz][2] = (var[ix, iy, iz + 1] - var[ix, iy, iz - 1]) * self.dxyzi / 2

    @ti.func
    def diverg_var(self, var, var_div):

        for ix, iy, iz in ti.ndrange(self.Nx - 1, self.Ny, self.Nz - 1):
            var_div[ix, iy, iz] = (var[ix + 1, iy, iz][0] - var[ix, iy, iz][0] +
                                   var[ix, iy + 1, iz][1] - var[ix, iy, iz][1] +
                                   var[ix, iy, iz + 1][2] - var[ix, iy, iz][2]) * self.dxyzi

    @ti.func
    def temp_fvm(self, cfd):

        for i in ti.grouped(cfd):
            cfd[i].temp_td = cfd[i].temp

        for ix, iy, iz in ti.ndrange((2, self.Nx - 2), (2, self.Ny - 2), (2, self.Nz - 2)):
            D_e, D_w = cfd[ix + 1, iy, iz].kappa * self.dxyz, cfd[ix - 1, iy, iz].kappa * self.dxyz
            D_n, D_s = cfd[ix, iy + 1, iz].kappa * self.dxyz, cfd[ix, iy - 1, iz].kappa * self.dxyz
            D_f, D_b = cfd[ix, iy, iz + 1].kappa * self.dxyz, cfd[ix, iy, iz - 1].kappa * self.dxyz

            F_e = cfd[ix + 1, iy, iz].rho_cp * cfd[ix + 1, iy, iz].vel[0] * self.dxyz ** 2
            F_w = cfd[ix - 1, iy, iz].rho_cp * cfd[ix - 1, iy, iz].vel[0] * self.dxyz ** 2
            F_n = cfd[ix, iy + 1, iz].rho_cp * cfd[ix, iy + 1, iz].vel[1] * self.dxyz ** 2
            F_s = cfd[ix, iy - 1, iz].rho_cp * cfd[ix, iy - 1, iz].vel[1] * self.dxyz ** 2
            F_f = cfd[ix, iy, iz + 1].rho_cp * cfd[ix, iy, iz + 1].vel[2] * self.dxyz ** 2
            F_b = cfd[ix, iy, iz - 1].rho_cp * cfd[ix, iy, iz - 1].vel[2] * self.dxyz ** 2

            a_e = D_e + 1.5 * ti.max(-F_e, 0) + 0.5 * ti.max(-F_w, 0)
            a_w = D_w + 1.5 * ti.max(F_w, 0) + 0.5 * ti.max(F_e, 0)
            a_n = D_n + 1.5 * ti.max(-F_n, 0) + 0.5 * ti.max(-F_s, 0)
            a_s = D_s + 1.5 * ti.max(F_s, 0) + 0.5 * ti.max(F_n, 0)
            a_f = D_f + 1.5 * ti.max(-F_f, 0) + 0.5 * ti.max(-F_b, 0)
            a_b = D_b + 1.5 * ti.max(F_b, 0) + 0.5 * ti.max(F_f, 0)

            ap0 = cfd[ix, iy, iz].rho_cp * self.dxyz ** 3 / self.dt
            a_p = (ap0 + D_e + D_w + D_n + D_s + D_f + D_b +
                   1.5 * (ti.max(F_e, 0) + ti.max(-F_w, 0) +
                          ti.max(F_n, 0) + ti.max(-F_s, 0) +
                          ti.max(F_f, 0) + ti.max(-F_b, 0)))

            temp_rhs_2 = -((cfd[ix, iy, iz].q_con_rad_vap * cfd[ix, iy, iz].leng_grad_alpha_1 -   ###
                            cfd[ix, iy, iz].q_laser) * cfd[ix, iy, iz].rho_cp / self.avg_rho_cp)

            b = (ap0 * cfd[ix, iy, iz].temp_td + temp_rhs_2 * self.dxyz ** 3 -
                 0.5 * (cfd[ix + 2, iy, iz].temp_td * ti.max(-F_e, 0) +
                        cfd[ix - 2, iy, iz].temp_td * ti.max(F_w, 0) +
                        cfd[ix, iy + 2, iz].temp_td * ti.max(-F_n, 0) +
                        cfd[ix, iy - 2, iz].temp_td * ti.max(F_s, 0) +
                        cfd[ix, iy, iz + 2].temp_td * ti.max(-F_f, 0) +
                        cfd[ix, iy, iz - 2].temp_td * ti.max(F_b, 0)))

            rhs = (a_e * cfd[ix + 1, iy, iz].temp_td + a_w * cfd[ix - 1, iy, iz].temp_td +
                   a_n * cfd[ix, iy + 1, iz].temp_td + a_s * cfd[ix, iy - 1, iz].temp_td +
                   a_f * cfd[ix, iy, iz + 1].temp_td + a_b * cfd[ix, iy, iz - 1].temp_td + b)

            cfd[ix, iy, iz].temp = tm.clamp(rhs / a_p, self.T_ambient, self.T_vapor)

        for ix, iy, iz, ex in ti.ndrange(self.Nx, self.Ny, self.Nz, 2):
            cfd[ix, iy, ex].temp = args.T_subplate
            cfd[ix, iy, self.Nz - 1 - ex].temp = cfd[ix, iy, self.Nz - 3].temp

    @ti.func
    def temp_terms(self, cfd, step):

        laser_x, laser_y = args.laser_info[step, 1], args.laser_info[step, 2]
        tmp_topsurface = self.top_surface - (self.layer_order - 2) * self.layer_thick

        for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):

            i = ix, iy, iz
            Pv_T = self.p0 * tm.exp(self.latent_v / self.gas_const * (1 / self.T_vapor - 1 / cfd[i].temp))

            cfd[i].q_laser = (self.laser_coeff *
                ti.exp(-2 * ((ix * self.dxyz - laser_x) ** 2 + (iy * self.dxyz - laser_y) ** 2) / self.r_beam ** 2) *
                ti.exp(-ti.abs(iz * self.dxyz - tmp_topsurface) / self.layer_thick))

            q_con = self.h_conv * (cfd[i].temp - self.T_ambient)
            q_rad = self.SB_const * self.emissivity * (cfd[i].temp ** 4 - self.T_ambient ** 4)
            q_vap = (cfd[i].alpha_1 * Pv_T * ti.sqrt(self.mo / (2 * math.pi * self.Kb * cfd[i].temp)) * self.latent_v)

            cfd[i].q_con_rad_vap = q_con + q_rad + q_vap

    @ti.func
    def solve_p_jacobi(self, rhs, pt, p):

        x0, x1 = 1, self.Nx - 1
        y0, y1 = 1, self.Ny - 1
        z0, z1 = 1, self.Nz - 1

        ti.loop_config(serialize=True)
        for _ in range(10):
            for ix, iy, iz in ti.ndrange((x0, x1), (y0, y1), (z0, z1)):
                a_e = self.dxyzi ** 2 if ix != x1 - 1 else 0.
                a_w = self.dxyzi ** 2 if ix != x0 else 0.
                a_n = self.dxyzi ** 2 if iy != y1 - 1 else 0.
                a_s = self.dxyzi ** 2 if iy != y0 else 0.
                a_f = self.dxyzi ** 2 if iz != z1 - 1 else 0.
                a_b = self.dxyzi ** 2 if iz != z0 else 0.
                a_p = - 1.0 * (a_e + a_w + a_n + a_s + a_f + a_b)
                pt[ix, iy, iz] = (rhs[ix, iy, iz] - a_e * p[ix + 1, iy, iz] - a_w * p[ix - 1, iy, iz]
                                                  - a_n * p[ix, iy + 1, iz] - a_s * p[ix, iy - 1, iz]
                                                  - a_f * p[ix, iy, iz + 1] - a_b * p[ix, iy, iz - 1]) / a_p

            for ix, iy, iz in ti.ndrange((x0, x1), (y0, y1), (z0, z1)):
                p[ix, iy, iz] = pt[ix, iy, iz]

        for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):
            p[0, iy, iz], p[self.Nx - 1, iy, iz] = p[1, iy, iz], p[self.Nx - 2, iy, iz]
            p[ix, 0, iz], p[ix, self.Ny - 1, iz] = p[ix, 1, iz], p[ix, self.Ny - 2, iz]
            p[ix, iy, 0], p[ix, iy, self.Nz - 1] = p[ix, iy, 1], p[ix, iy, self.Nz - 2]

    @ti.func
    def advect_fvm(self, vel, vel_star, rho, visco, F_total):

        for ix, iy, iz in ti.ndrange((1, self.Nx - 1), (1, self.Ny - 1), (1, self.Nz - 1)):

            # x direction
            vh = (vel[ix - 1, iy, iz][1] + vel[ix - 1, iy + 1, iz][1] + vel[ix, iy, iz][1] + vel[ix, iy + 1, iz][1]) / 4
            wh = (vel[ix - 1, iy, iz][2] + vel[ix - 1, iy, iz + 1][2] + vel[ix, iy, iz][2] + vel[ix, iy, iz + 1][2]) / 4

            dudx = (vel[ix, iy, iz][0] - vel[ix - 1, iy, iz][0]) * self.dxyzi if vel[ix, iy, iz][0] > 0 else \
                   (vel[ix + 1, iy, iz][0] - vel[ix, iy, iz][0]) * self.dxyzi

            dudy = (vel[ix, iy, iz][0] - vel[ix, iy - 1, iz][0]) * self.dxyzi if vh > 0 else \
                   (vel[ix, iy + 1, iz][0] - vel[ix, iy, iz][0]) * self.dxyzi

            dudz = (vel[ix, iy, iz][0] - vel[ix, iy, iz - 1][0]) * self.dxyzi if wh > 0 else \
                   (vel[ix, iy, iz + 1][0] - vel[ix, iy, iz][0]) * self.dxyzi

            laplace_u = (vel[ix - 1, iy, iz][0] + vel[ix + 1, iy, iz][0] - 2 * vel[ix, iy, iz][0] +
                         vel[ix, iy - 1, iz][0] + vel[ix, iy + 1, iz][0] - 2 * vel[ix, iy, iz][0] +
                         vel[ix, iy, iz - 1][0] + vel[ix, iy, iz + 1][0] - 2 * vel[ix, iy, iz][0]) * self.dxyzi ** 2

            vel_star[ix, iy, iz][0] = (vel[ix, iy, iz][0] + self.dt / rho[ix, iy, iz] * (laplace_u * visco[ix, iy, iz] -
                        rho[ix, iy, iz] * (vel[ix, iy, iz][0] * dudx + vh * dudy + wh * dudz) + F_total[ix, iy, iz][0]))

            # y direction
            uh = (vel[ix, iy - 1, iz][0] + vel[ix + 1, iy - 1, iz][0] + vel[ix, iy, iz][0] + vel[ix + 1, iy, iz][0]) / 4
            wh = (vel[ix, iy - 1, iz][2] + vel[ix, iy - 1, iz + 1][2] + vel[ix, iy, iz][2] + vel[ix, iy, iz + 1][2]) / 4

            dvdx = (vel[ix, iy, iz][1] - vel[ix - 1, iy, iz][1]) * self.dxyzi if uh > 0 else \
                   (vel[ix + 1, iy, iz][1] - vel[ix, iy, iz][1]) * self.dxyzi

            dvdy = (vel[ix, iy, iz][1] - vel[ix, iy - 1, iz][1]) * self.dxyzi if vel[ix, iy, iz][1] > 0 else \
                   (vel[ix, iy + 1, iz][1] - vel[ix, iy, iz][1]) * self.dxyzi

            dvdz = (vel[ix, iy, iz][1] - vel[ix, iy, iz - 1][1]) * self.dxyzi if wh > 0 else \
                   (vel[ix, iy, iz + 1][1] - vel[ix, iy, iz][1]) * self.dxyzi

            laplace_v = (vel[ix - 1, iy, iz][1] + vel[ix + 1, iy, iz][1] - 2 * vel[ix, iy, iz][1] +
                         vel[ix, iy - 1, iz][1] + vel[ix, iy + 1, iz][1] - 2 * vel[ix, iy, iz][1] +
                         vel[ix, iy, iz - 1][1] + vel[ix, iy, iz + 1][1] - 2 * vel[ix, iy, iz][1]) * self.dxyzi ** 2

            vel_star[ix, iy, iz][1] = (vel[ix, iy, iz][1] + self.dt / rho[ix, iy, iz] * (laplace_v * visco[ix, iy, iz] -
                        rho[ix, iy, iz] * (uh * dvdx + vel[ix, iy, iz][1] * dvdy + wh * dvdz) + F_total[ix, iy, iz][1]))

            # z direction
            uh = (vel[ix, iy, iz - 1][0] + vel[ix + 1, iy, iz - 1][0] + vel[ix, iy, iz][0] + vel[ix + 1, iy, iz][0]) / 4
            vh = (vel[ix, iy, iz - 1][1] + vel[ix, iy + 1, iz - 1][1] + vel[ix, iy, iz][1] + vel[ix, iy + 1, iz][1]) / 4

            dwdx = (vel[ix, iy, iz][2] - vel[ix - 1, iy, iz][2]) * self.dxyzi if uh > 0 else \
                   (vel[ix + 1, iy, iz][2] - vel[ix, iy, iz][2]) * self.dxyzi

            dwdy = (vel[ix, iy, iz][2] - vel[ix, iy - 1, iz][2]) * self.dxyzi if vh > 0 else \
                   (vel[ix, iy + 1, iz][2] - vel[ix, iy, iz][2]) * self.dxyzi

            dwdz = (vel[ix, iy, iz][2] - vel[ix, iy, iz - 1][2]) * self.dxyzi if vel[ix, iy, iz][2] > 0 else \
                   (vel[ix, iy, iz + 1][2] - vel[ix, iy, iz][2]) * self.dxyzi

            laplace_w = (vel[ix - 1, iy, iz][2] + vel[ix + 1, iy, iz][2] - 2 * vel[ix, iy, iz][2] +
                         vel[ix, iy - 1, iz][2] + vel[ix, iy + 1, iz][2] - 2 * vel[ix, iy, iz][2] +
                         vel[ix, iy, iz - 1][2] + vel[ix, iy, iz + 1][2] - 2 * vel[ix, iy, iz][2]) * self.dxyzi ** 2

            vel_star[ix, iy, iz][2] = (vel[ix, iy, iz][2] + self.dt / rho[ix, iy, iz] * (laplace_w * visco[ix, iy, iz] -
                        rho[ix, iy, iz] * (uh * dwdx + vh * dwdy + vel[ix, iy, iz][2] * dwdz) + F_total[ix, iy, iz][2]))

    @ti.func
    def force_cal(self, cfd):

        self.grad_var(cfd.temp, cfd.grad_temp)
        self.grad_var(cfd.alpha_1, cfd.grad_alpha_1)

        # normal direction of interface
        for i in ti.grouped(cfd):
            cfd[i].leng_grad_alpha_1 = tm.length(cfd[i].grad_alpha_1)
            cfd[i].norm_grad_alpha_1 = tm.normalize(cfd[i].grad_alpha_1)
            if cfd[i].leng_grad_alpha_1 <= 1e-10:
                cfd[i].leng_grad_alpha_1 = 1e-10
                cfd[i].norm_grad_alpha_1 = ti.Vector([0., 0., 0.])

        self.diverg_var(cfd.norm_grad_alpha_1, cfd.Kc)

        for i in ti.grouped(cfd):

            # second term in the bracket of Marangoni force
            F_marangoni_n_grad_temp = tm.dot(cfd[i].norm_grad_alpha_1, cfd[i].grad_temp)

            # coefficient for the recoil force
            Pv_T = self.p0 * tm.exp(self.latent_v / self.gas_const * (1 / self.T_vapor - 1 / cfd[i].temp))

            for d in range(3):

                # gravitational force, the gravitational force only work along z axis
                Force_0 = -cfd[i].rho * self.gra_const * (d == 2)

                # buoyancy force
                Force_1 = -Force_0 * self.beta * (cfd[i].temp - self.T_ambient)

                # mushy force
                Force_2 = (-cfd[i].alpha_1 * cfd[i].rho * self.K_C * cfd[i].vel[d] *
                          ((1 - cfd[i].gamma) ** 2 / (cfd[i].gamma ** 3 + self.C_K)))

                # surface tension force
                Force_3 = -self.sigma * cfd[i].Kc * cfd[i].norm_grad_alpha_1[d]

                # Marangoni force
                Force_4 = self.dsigma_dT * (cfd[i].grad_temp[d] - cfd[i].norm_grad_alpha_1[d] * F_marangoni_n_grad_temp)

                # recoil force
                Force_5 = Pv_T * cfd[i].norm_grad_alpha_1[d]

                cfd[i].F_total[d] = (Force_0 + Force_1 + Force_2 + (Force_3 + Force_4 + Force_5) *
                                     cfd[i].leng_grad_alpha_1 * 2 * cfd[i].rho / (self.rho_metal + self.rho_gas) *   ###
                                     cfd[i].alpha_1 * cfd[i].gamma)

    @ti.kernel
    def args_update(self, cfd: ti.template()):

        z0 = ti.max(0, args.Nz - args.reduced_Nz)
        for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):
            args.temp[ix, iy, z0 + iz] = cfd[ix, iy, iz].temp

            args.zeta[ix, iy, z0 + iz] = cfd[ix, iy, iz].alpha_1
            if tm.isnan(args.zeta[ix, iy, z0 + iz]):
                args.zeta[ix, iy, z0 + iz] = 0.

        for ix, iy, ex in ti.ndrange(self.Nx, self.Ny, 2):
            args.temp[ix, iy, ex] = self.T_subplate

    @ti.kernel
    def vof_update(self, cfd: ti.template()):

        for i in ti.grouped(cfd):
            cfd[i].alpha_1_td = cfd[i].alpha_1

        for ix, iy, iz in ti.ndrange((1, self.Nx - 1), (1, self.Ny - 1), (1, self.Nz - 1)):

            a_e = self.dt * cfd[ix + 1, iy, iz].vel[0] * cfd[ix, iy, iz].alpha_1_td if cfd[ix + 1, iy, iz].vel[0] > 0 \
             else self.dt * cfd[ix + 1, iy, iz].vel[0] * cfd[ix + 1, iy, iz].alpha_1_td

            a_w = self.dt * cfd[ix, iy, iz].vel[0] * cfd[ix - 1, iy, iz].alpha_1_td if cfd[ix, iy, iz].vel[0] > 0 \
             else self.dt * cfd[ix, iy, iz].vel[0] * cfd[ix, iy, iz].alpha_1_td

            a_n = self.dt * cfd[ix, iy + 1, iz].vel[1] * cfd[ix, iy, iz].alpha_1_td if cfd[ix, iy + 1, iz].vel[1] > 0 \
             else self.dt * cfd[ix, iy + 1, iz].vel[1] * cfd[ix, iy + 1, iz].alpha_1_td

            a_s = self.dt * cfd[ix, iy, iz].vel[1] * cfd[ix, iy - 1, iz].alpha_1_td if cfd[ix, iy, iz].vel[1] > 0 \
             else self.dt * cfd[ix, iy, iz].vel[1] * cfd[ix, iy, iz].alpha_1_td

            a_f = self.dt * cfd[ix, iy, iz + 1].vel[2] * cfd[ix, iy, iz].alpha_1_td if cfd[ix, iy, iz + 1].vel[2] > 0 \
             else self.dt * cfd[ix, iy, iz + 1].vel[2] * cfd[ix, iy, iz + 1].alpha_1_td

            a_b = self.dt * cfd[ix, iy, iz].vel[2] * cfd[ix, iy, iz - 1].alpha_1_td if cfd[ix, iy, iz].vel[2] > 0 \
             else self.dt * cfd[ix, iy, iz].vel[2] * cfd[ix, iy, iz].alpha_1_td

            rhs = (self.p0 * ti.exp(self.latent_v / self.gas_const * (1 / self.T_vapor - 1 / cfd[ix, iy, iz].temp)) *
                  (self.mo / (2 * math.pi * self.Kb * cfd[ix, iy, iz].temp)) ** 0.5 / self.rho_gas)

            cfd[ix, iy, iz].alpha_1 += -(a_e - a_w + a_n - a_s + a_f - a_b) * self.dxyzi - self.dt * rhs

            cfd[ix, iy, iz].alpha_1 = tm.clamp(cfd[ix, iy, iz].alpha_1, 0., 1.)

        for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):

            cfd[0, iy, iz].alpha_1 = cfd[1, iy, iz].alpha_1
            cfd[ix, 0, iz].alpha_1 = cfd[ix, 1, iz].alpha_1
            cfd[ix, iy, 0].alpha_1 = cfd[ix, iy, 1].alpha_1

            cfd[self.Nx - 1, iy, iz].alpha_1 = cfd[self.Nx - 2, iy, iz].alpha_1
            cfd[ix, self.Ny - 1, iz].alpha_1 = cfd[ix, self.Ny - 2, iz].alpha_1
            cfd[ix, iy, self.Nz - 1].alpha_1 = cfd[ix, iy, self.Nz - 2].alpha_1

    @ti.kernel
    def temp_update(self, cfd: ti.template(), step: int):

        # Ref: https://doi.org/10.2514/3.11010 (FVM)
        self.temp_terms(cfd, step)
        self.temp_fvm(cfd)

    @ti.kernel
    def vel_update(self, cfd: ti.template()):

        self.force_cal(cfd)
        self.advect_fvm(cfd.vel, cfd.vel_star, cfd.rho, cfd.visco, cfd.F_total)

        # the correction step to ensure no divergence
        self.diverg_var(cfd.vel_star, cfd.vel_star_div)

        for i in ti.grouped(cfd):
            cfd[i].possion_p_rhs = cfd[i].rho * cfd[i].vel_star_div / self.dt

        self.solve_p_jacobi(cfd.possion_p_rhs, cfd.pt, cfd.p)

        # velocity field calculation
        for ix, iy, iz in ti.ndrange((1, self.Nx - 1), (1, self.Ny - 1), (1, self.Nz - 1)):

            # x direction
            r = (cfd[ix, iy, iz].rho + cfd[ix - 1, iy, iz].rho) / 2
            grad_p = (cfd[ix, iy, iz].p - cfd[ix - 1, iy, iz].p) * self.dxyzi
            cfd[ix, iy, iz].vel[0] = cfd[ix, iy, iz].vel_star[0] - self.dt * grad_p / r
            if cfd[ix, iy, iz].vel[0] * self.dt > 0.25 * self.dxyz:
                self.iflag[0] = 1
                print('x', ix, iy, iz, cfd[ix, iy, iz].vel[0])

            # y direction
            r = (cfd[ix, iy, iz].rho + cfd[ix, iy - 1, iz].rho) / 2
            grad_p = (cfd[ix, iy, iz].p - cfd[ix, iy - 1, iz].p) * self.dxyzi
            cfd[ix, iy, iz].vel[1] = cfd[ix, iy, iz].vel_star[1] - self.dt * grad_p / r
            if cfd[ix, iy, iz].vel[1] * self.dt > 0.25 * self.dxyz:
                self.iflag[0] = 1
                print('y', ix, iy, iz, cfd[ix, iy, iz].vel[1])

            # z direction
            r = (cfd[ix, iy, iz].rho + cfd[ix, iy, iz - 1].rho) / 2
            grad_p = (cfd[ix, iy, iz].p - cfd[ix, iy, iz - 1].p) * self.dxyzi
            cfd[ix, iy, iz].vel[2] = cfd[ix, iy, iz].vel_star[2] - self.dt * grad_p / r
            if cfd[ix, iy, iz].vel[2] * self.dt > 0.25 * self.dxyz:
                self.iflag[0] = 1
                print('z', ix, iy, iz, cfd[ix, iy, iz].vel[2])

        for ix, iy, iz, d in ti.ndrange(self.Nx, self.Ny, self.Nz, 3):

            cfd[0, iy, iz].vel[d], cfd[self.Nx - 1, iy, iz].vel[d] = 0., 0.
            cfd[ix, 0, iz].vel[d], cfd[ix, self.Ny - 1, iz].vel[d] = 0., 0.
            cfd[ix, iy, 0].vel[d], cfd[ix, iy, self.Nz - 1].vel[d] = 0., 0.
            cfd[1, iy, iz].vel[d], cfd[self.Nx - 2, iy, iz].vel[d] = 0., 0.
            cfd[ix, 1, iz].vel[d], cfd[ix, self.Ny - 2, iz].vel[d] = 0., 0.
            cfd[ix, iy, 1].vel[d], cfd[ix, iy, self.Nz - 2].vel[d] = 0., 0.

    @ti.kernel
    def properties_update(self, cfd: ti.template()):

        for i in ti.grouped(cfd):

            alpha_1, temp = cfd[i].alpha_1, cfd[i].temp
            cfd[i].rho = alpha_1 * self.rho_metal + (1 - alpha_1) * self.rho_gas

            r = tm.clamp((temp - self.T_solid) / (self.T_liquid - self.T_solid), 0., 1.)
            fl = alpha_1 * r ** 3 * (6 * r ** 2 - 15 * r + 10)
            cfd[i].cp = (alpha_1 - fl) * self.cp_solid + fl * self.cp_liquid + (1 - alpha_1) * self.cp_gas
            cfd[i].kappa = (alpha_1 - fl) * self.kappa_solid + fl * self.kappa_liquid + (1 - alpha_1) * self.kappa_gas
            cfd[i].visco = (alpha_1 - fl) * self.visco_solid + fl * self.visco_liquid + (1 - alpha_1) * self.visco_gas
            cfd[i].rho_cp = (self.rho_metal * ((alpha_1 - fl) * self.cp_solid + fl * self.cp_liquid) +
                             self.rho_gas * (1 - alpha_1) * self.cp_gas)

            cfd[i].gamma = tm.clamp((temp - self.T_solid) / (self.T_liquid - self.T_solid), 0., 1.)

    @ti.kernel
    def subarea_init(self, cfd: ti.template(), step: int):

        z0 = ti.max(0, args.Nz - args.reduced_Nz)

        if int(args.laser_info[step, 3]):
            for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):
                cfd[ix, iy, iz - z0].vel = ti.Vector([0., 0., 0.])
                cfd[ix, iy, iz - z0].p = 0.

        for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):
            cfd[ix, iy, iz].alpha_1 = args.zeta[ix, iy, iz + z0]

        if step == 0:
            for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, args.total_Nz):
                args.temp[ix, iy, iz] = self.T_ambient

            for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):
                cfd[ix, iy, iz].temp = self.T_ambient

        else:
            if self.layer_switch[0]:
                for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):
                    cfd[ix, iy, iz - z0].vel = ti.Vector([0., 0., 0.])

                for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, args.Nz_old):
                    args.temp[ix, iy, iz] = args.temp0[ix, iy, iz]

                self.layer_switch[0] = 0

            for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):
                cfd[ix, iy, iz].temp = args.temp[ix, iy, iz + z0]

    def temp_solver(self, step):

        self.subarea_init(self.cfd, step)
        self.properties_update(self.cfd)
        self.vel_update(self.cfd)
        self.temp_update(self.cfd, step)
        self.vof_update(self.cfd)
        self.args_update(self.cfd)

        # for debug --------------------
        args.iflag_cfd = self.iflag.to_numpy()


@ti.data_oriented
class PfmFunc:

    def __init__(self):

        self.dt, self.dxyz = args.dt_PFM, args.dxyz
        self.Nx, self.Ny = args.Nx, args.Ny
        self.Nz = args.reduced_Nz + args.layer_Nz * 0  # reduced area size in z direction ////////////////////

        self.num_oris = args.num_oris

        self.T_solid, self.T_liquid, self.T_vapor = args.T_solid, args.T_liquid, args.T_vapor
        self.T_subplate, self.T_ambient = args.T_subplate, args.T_ambient

        self.L0, self.Qg, self.L_p, self.L_g = args.L0, args.Qg, args.L_p, args.L_g
        self.delta_fp, self.delta_fg = args.delta_fp, args.delta_fg
        self.sigma_p, self.sigma_g0 = args.sigma_p, args.sigma_g0
        self.gamma, self.degree_aniso = args.gamma, args.degree_aniso
        self.latent_s, self.latent_v = args.latent_s, args.latent_v

        self.gas_const = args.gas_const

        self.Lp_T = self.L0 * ti.exp(-self.Qg / (self.gas_const * self.T_liquid))
        self.m_p = 3 / 4 * self.sigma_p / (self.delta_fp * self.L_p)
        self.m_g = 3 / 4 * self.sigma_g0 / (self.delta_fg * self.L_g)
        self.kappa_p = 3 / 4 * self.sigma_p * self.L_p

        self.pfm_var = ti.Struct.field({
            'phi': float,
            'zeta': float,
            'temp': float,

            'phi_td': float,
            'zeta_td': float,

            'df_dzeta': float,
            'df_dphi': float,

            'Lp_T': float,
            'Lg_T': float,
            'kappa_g': float,

            'ori_id': int,
            'sum_phi2': float,
        }, shape=(self.Nx, self.Ny, self.Nz, self.num_oris))

        self.phi_xyz = ti.field(dtype=float, shape=(self.Nz, self.Ny, self.Nx, self.num_oris))
        self.edges = ti.field(dtype=float, shape=(self.Nz, self.Ny, self.Nx, self.num_oris, 3))
        self.directions = ti.field(dtype=float, shape=(self.Nx * self.Ny * self.Nz, self.num_oris, 3))

        # ---------for debug------------
        self.iflag = ti.field(dtype=ti.int32, shape=1)

    @ti.kernel
    def args_update(self, pfm_var: ti.template()):

        # 0: phase fraction; 1 : -2: grain faction; -1: orientation_id
        z0 = ti.max(0, args.Nz - args.reduced_Nz)
        for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):
            args.zeta[ix, iy, z0 + iz] = pfm_var[ix, iy, iz, 0].zeta

        for ix, iy, iz, g in ti.ndrange(self.Nx, self.Ny, self.Nz, self.num_oris):
            args.phi[ix, iy, z0 + iz, g] = pfm_var[ix, iy, iz, g].phi
            args.phi[ix, iy, z0 + iz, self.num_oris] = pfm_var[ix, iy, iz, 0].ori_id

        # ---------for debug------------
        for ix, iy in ti.ndrange(self.Nx, self.Ny):
            if pfm_var[ix, iy, self.Nz - 3, 0].zeta >= 0.5:
                self.iflag[0] = 1

    @ti.kernel
    def set_bc(self, pfm_var: ti.template()):

        for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):

            pfm_var[0, iy, iz, 0].zeta = pfm_var[1, iy, iz, 0].zeta
            pfm_var[ix, 0, iz, 0].zeta = pfm_var[ix, 1, iz, 0].zeta
            pfm_var[ix, iy, 0, 0].zeta = pfm_var[ix, iy, 1, 0].zeta

            pfm_var[self.Nx - 1, iy, iz, 0].zeta = pfm_var[self.Nx - 2, iy, iz, 0].zeta
            pfm_var[ix, self.Ny - 1, iz, 0].zeta = pfm_var[ix, self.Ny - 2, iz, 0].zeta
            pfm_var[ix, iy, self.Nz - 1, 0].zeta = pfm_var[ix, iy, self.Nz - 2, 0].zeta
        for ix, iy, iz, g in ti.ndrange(self.Nx, self.Ny, self.Nz, self.num_oris):
            pfm_var[0, iy, iz, g].phi = pfm_var[1, iy, iz, g].phi
            pfm_var[ix, 0, iz, g].phi = pfm_var[ix, 1, iz, g].phi
            pfm_var[ix, iy, 0, g].phi = pfm_var[ix, iy, 1, g].phi

            pfm_var[self.Nx - 1, iy, iz, g].phi = pfm_var[self.Nx - 2, iy, iz, g].phi
            pfm_var[ix, self.Ny - 1, iz, g].phi = pfm_var[ix, self.Ny - 2, iz, g].phi
            pfm_var[ix, iy, self.Nz - 1, g].phi = pfm_var[ix, iy, self.Nz - 2, g].phi

        for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):
            pfm_var[ix, iy, iz, 0].ori_id = 0
            for g in range(self.num_oris):
                if pfm_var[ix, iy, iz, g].phi >= 0.5:
                    pfm_var[ix, iy, iz, 0].ori_id = g + 1

    @ti.kernel
    def zeta_phi_update(self, pfm_var: ti.template()):

        for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):
            pfm_var[ix, iy, iz, 0].zeta_td = pfm_var[ix, iy, iz, 0].zeta

            pfm_var[ix, iy, iz, 0].ori_id = 0  ####
            for g in range(self.num_oris):
                pfm_var[ix, iy, iz, g].phi_td = pfm_var[ix, iy, iz, g].phi

        # -----zeta update-----
        for ix, iy, iz in ti.ndrange((1, self.Nx - 1), (1, self.Ny - 1), (1, self.Nz - 1)):

            a_e = -pfm_var[ix + 1, iy, iz, 0].Lp_T * (-self.kappa_p) * self.dxyz
            a_w = -pfm_var[ix - 1, iy, iz, 0].Lp_T * (-self.kappa_p) * self.dxyz
            a_n = -pfm_var[ix, iy + 1, iz, 0].Lp_T * (-self.kappa_p) * self.dxyz
            a_s = -pfm_var[ix, iy - 1, iz, 0].Lp_T * (-self.kappa_p) * self.dxyz
            a_f = -pfm_var[ix, iy, iz + 1, 0].Lp_T * (-self.kappa_p) * self.dxyz
            a_b = -pfm_var[ix, iy, iz - 1, 0].Lp_T * (-self.kappa_p) * self.dxyz
            ap0 = self.dxyz ** 3 / self.dt
            ap = ap0 + a_e + a_w + a_n + a_s + a_f + a_b

            rhs = (ap0 * pfm_var[ix, iy, iz, 0].zeta_td -
                   pfm_var[ix, iy, iz, 0].Lp_T * pfm_var[ix, iy, iz, 0].df_dzeta * self.dxyz ** 3 +
                   a_e * pfm_var[ix + 1, iy, iz, 0].zeta_td + a_w * pfm_var[ix - 1, iy, iz, 0].zeta_td +
                   a_n * pfm_var[ix, iy + 1, iz, 0].zeta_td + a_s * pfm_var[ix, iy - 1, iz, 0].zeta_td +
                   a_f * pfm_var[ix, iy, iz + 1, 0].zeta_td + a_b * pfm_var[ix, iy, iz - 1, 0].zeta_td)

            pfm_var[ix, iy, iz, 0].zeta = tm.clamp(rhs / ap, 0., 1.)

        # -----phi update-----
        for ix, iy, iz, g in ti.ndrange((1, self.Nx - 1), (1, self.Ny - 1), (1, self.Nz - 1), self.num_oris):
            temp, zeta0 = pfm_var[ix, iy, iz, 0].temp, pfm_var[ix, iy, iz, 0].zeta_td
            zeta_e, zeta_w = pfm_var[ix + 1, iy, iz, 0].zeta_td, pfm_var[ix - 1, iy, iz, 0].zeta_td
            zeta_n, zeta_s = pfm_var[ix, iy + 1, iz, 0].zeta_td, pfm_var[ix, iy - 1, iz, 0].zeta_td
            zeta_f, zeta_b = pfm_var[ix, iy, iz + 1, 0].zeta_td, pfm_var[ix, iy, iz - 1, 0].zeta_td

            a_e = -pfm_var[ix + 1, iy, iz, 0].Lg_T * (-pfm_var[ix + 1, iy, iz, g].kappa_g) * self.dxyz * zeta_e
            a_w = -pfm_var[ix - 1, iy, iz, 0].Lg_T * (-pfm_var[ix - 1, iy, iz, g].kappa_g) * self.dxyz * zeta_w
            a_n = -pfm_var[ix, iy + 1, iz, 0].Lg_T * (-pfm_var[ix, iy + 1, iz, g].kappa_g) * self.dxyz * zeta_n
            a_s = -pfm_var[ix, iy - 1, iz, 0].Lg_T * (-pfm_var[ix, iy - 1, iz, g].kappa_g) * self.dxyz * zeta_s
            a_f = -pfm_var[ix, iy, iz + 1, 0].Lg_T * (-pfm_var[ix, iy, iz + 1, g].kappa_g) * self.dxyz * zeta_f
            a_b = -pfm_var[ix, iy, iz - 1, 0].Lg_T * (-pfm_var[ix, iy, iz - 1, g].kappa_g) * self.dxyz * zeta_b
            ap0 = self.dxyz ** 3 / self.dt
            ap = ap0 + a_e + a_w + a_n + a_s + a_f + a_b

            rhs = (ap0 * pfm_var[ix, iy, iz, g].phi_td -
                   pfm_var[ix, iy, iz, 0].Lg_T * pfm_var[ix, iy, iz, g].df_dphi * self.dxyz ** 3 * zeta0 +
                   a_e * pfm_var[ix + 1, iy, iz, g].phi_td + a_w * pfm_var[ix - 1, iy, iz, g].phi_td +
                   a_n * pfm_var[ix, iy + 1, iz, g].phi_td + a_s * pfm_var[ix, iy - 1, iz, g].phi_td +
                   a_f * pfm_var[ix, iy, iz + 1, g].phi_td + a_b * pfm_var[ix, iy, iz - 1, g].phi_td)

            r = tm.clamp((self.T_liquid - temp) / (self.T_liquid - self.T_solid), 0., 1.)
            ratio = r ** 3 * (6 * r ** 2 - 15 * r + 10)
            region = (self.T_solid <= temp <= self.T_liquid)
            phi_noise = (5e-04 + 1e-04 * (ti.random(float) - 0.5)) * ratio ** 1 * region * zeta0

            _phi = tm.clamp(rhs / ap + phi_noise, 0., 1.)
            pfm_var[ix, iy, iz, g].phi = tm.min(_phi, zeta0 * ratio)

            if pfm_var[ix, iy, iz, g].phi >= 0.5:
                pfm_var[ix, iy, iz, 0].ori_id = g + 1

        ###-------------------
        pfm_var.sum_phi2.fill(0)
        for ix, iy, iz, g in ti.ndrange(self.Nx, self.Ny, self.Nz, self.num_oris):
            pfm_var[ix, iy, iz, 0].sum_phi2 += pfm_var[ix, iy, iz, g].phi ** 2

    @ti.kernel
    def df_cal(self, pfm_var: ti.template()):

        # ------ df_dzeta -------
        for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):

            temp, zeta = pfm_var[ix, iy, iz, 0].temp, pfm_var[ix, iy, iz, 0].zeta
            r0 = tm.clamp((self.T_liquid - temp) / (self.T_liquid - self.T_solid), 0., 1.)
            r = r0 ** 3 * (6 * r0 ** 2 - 15 * r0 + 10)

            s0 = tm.log(self.latent_s) * (temp - self.T_ambient) / self.T_vapor
            l0 = tm.log(self.latent_v) * (temp - self.T_ambient) / self.T_vapor
            delta_fp = r * s0 + (1 - r) * l0

            ap, bp, cp = delta_fp, delta_fp * (3 * 1 + 0.5), delta_fp * (2 * 1 + 0.5) ###
            pfm_var[ix, iy, iz, 0].df_dzeta = self.m_p * (ap * zeta - bp * zeta ** 2 + cp * zeta ** 3)

        # ------ df_dphi --------
        for ix, iy, iz, g in ti.ndrange(self.Nx, self.Ny, self.Nz, self.num_oris):

            phi, temp = pfm_var[ix, iy, iz, g].phi, pfm_var[ix, iy, iz, 0].temp
            delta_fg = tm.log(self.latent_s) * (temp - self.T_ambient) / self.T_ambient   ###

            ag, bg, cg = -delta_fg, delta_fg, 2 * delta_fg
            _term = 0.
            for gg in range(self.num_oris):
                _term += phi * pfm_var[ix, iy, iz, gg].phi ** 2
            _term -= phi ** 3
            pfm_var[ix, iy, iz, g].df_dphi = self.m_g * (ag * phi + bg * phi ** 3 + cg * _term)

    @ti.kernel
    def anisotropy_update(self, phi: ti.template(), kappa_g: ti.template()):

        Nx, Ny, Nz, num_oris = self.Nx, self.Ny, self.Nz, self.num_oris

        for ix0, iy0, iz0, g0 in ti.ndrange(Nx, Ny, Nz, num_oris):

            index = ix0 * Ny * Nz * num_oris + iy0 * Nz * num_oris + iz0 * num_oris + g0
            iz = index // (Ny * Nx * num_oris)
            iy = (index - iz * Ny * Nx * num_oris) // (Nx * num_oris)
            ix = (index - iz * Ny * Nx * num_oris - iy * Nx * num_oris) // num_oris
            g = index - iz * Ny * Nx * num_oris - iy * Nx * num_oris - ix * num_oris

            self.phi_xyz[iz, iy, ix, g] = phi[ix0, iy0, iz0, g0]

        for ix, iy, iz, g in ti.ndrange(Nx, Ny, Nz, num_oris):

            self.edges[iz, iy, ix, g, 0] = (self.phi_xyz[iz, iy, ix + (ix < Nx - 1), g] -
                                            self.phi_xyz[iz, iy, ix - (ix > 0), g])
            self.edges[iz, iy, ix, g, 1] = (self.phi_xyz[iz, iy + (iy < Ny - 1), ix, g] -
                                            self.phi_xyz[iz, iy - (iy > 0), ix, g])
            self.edges[iz, iy, ix, g, 2] = (self.phi_xyz[iz + (iz < Nz - 1), iy, ix, g] -
                                            self.phi_xyz[iz - (iz > 0), iy, ix, g])

        for iz0, iy0, ix0, g, d in ti.ndrange(Nz, Ny, Nx, num_oris, 3):

            index = iz0 * Ny * Nx * num_oris * 3 + iy0 * Nx * num_oris * 3 + ix0 * num_oris * 3 + g * 3 + d
            ix = index // (num_oris * 3)
            iy = (index - ix * num_oris * 3) // 3
            iz = index - ix * num_oris * 3 - iy * 3

            self.directions[ix, iy, iz] = self.edges[iz0, iy0, ix0, g, d]

        for ix, iy, iz, g in ti.ndrange(Nx, Ny, Nz, num_oris):

            angles = math.pi
            i = ix * Ny * Nz + iy * Nz + iz

            for d in range(3):

                go_Vec = ti.Vector([args.grain_ori[d, g, 0], args.grain_ori[d, g, 1], args.grain_ori[d, g, 2]])
                ed_Vec = ti.Vector([self.directions[i, g, 0], self.directions[i, g, 1], self.directions[i, g, 2]])

                cosines, demo = tm.dot(go_Vec, ed_Vec), tm.length(ed_Vec)
                cosines0 = 0. if demo == 0 else cosines / demo
                tmp_angle0 = tm.acos(cosines0)
                tmp_angle0 = tmp_angle0 if tmp_angle0 < math.pi / 2 else math.pi - tmp_angle0
                angles = tmp_angle0 if tmp_angle0 < angles else angles

            aniso_term = 1. + args.degree_aniso * (tm.cos(angles) ** 4 + tm.sin(angles) ** 4)
            kappa_g[ix, iy, iz, g] = 3 / 4 * self.L_g * self.sigma_g0 * aniso_term

    @ti.kernel
    def simuparameter_update(self, pfm_var: ti.template()):

        for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):
            temp = pfm_var[ix, iy, iz, 0].temp

            rt = tm.exp((temp - self.T_vapor) / self.T_ambient * 2) * 2 * 16

            pfm_var[ix, iy, iz, 0].Lp_T = self.Lp_T * rt
            pfm_var[ix, iy, iz, 0].Lg_T = (self.L0 * (temp / self.T_ambient) ** 0.7 *
                                           ti.exp(- self.Qg / (self.gas_const * temp))) * 1

    @ti.kernel
    def ti_init(self, temp: ti.template(), zeta: ti.template(), phi: ti.template()):

        z0 = ti.max(0, args.Nz - args.reduced_Nz)
        for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):

            temp[ix, iy, iz, 0] = args.temp[ix, iy, iz + z0]
            zeta[ix, iy, iz, 0] = args.zeta[ix, iy, iz + z0]

            for g in ti.static(range(self.num_oris)):
                phi[ix, iy, iz, g] = args.phi[ix, iy, iz + z0, g]

    def AC_solver(self):

        self.ti_init(self.pfm_var.temp, self.pfm_var.zeta, self.pfm_var.phi)
        self.simuparameter_update(self.pfm_var)
        self.anisotropy_update(self.pfm_var.phi, self.pfm_var.kappa_g)
        self.df_cal(self.pfm_var)
        self.zeta_phi_update(self.pfm_var)
        self.set_bc(self.pfm_var)
        self.args_update(self.pfm_var)

        # for debug --------------------
        args.iflag_pfm = self.iflag.to_numpy()
        args.sum_phi2 = self.pfm_var.sum_phi2.to_numpy()   ###########


@ti.data_oriented
class track_coolingdown:

    def __init__(self):

        self.num_oris = args.num_oris
        self.dt, self.dxyz = args.dt_PFM, args.dxyz
        self.Nx, self.Ny, self.Nz = args.Nx, args.Ny, args.reduced_Nz  # reduced area size in z direction

        self.rho_metal, self.rho_gas = args.rho_metal, args.rho_gas
        self.cp_solid, self.cp_liquid, self.cp_gas = args.cp_solid, args.cp_liquid, args.cp_gas
        self.kappa_solid, self.kappa_liquid, self.kappa_gas = args.kappa_solid, args.kappa_liquid, args.kappa_gas

        self.avg_rho_cp = (self.rho_metal * self.cp_solid + self.rho_gas * self.cp_gas) / 2

        self.T_solid, self.T_liquid, self.T_vapor = args.T_solid, args.T_liquid, args.T_vapor
        self.T_ambient = args.T_ambient

        self.mo, self.Kb, self.p0 = args.mo, args.Kb, args.p0
        self.latent_s, self.latent_v, self.gas_const = args.latent_s, args.latent_v, args.gas_const
        self.h_conv, self.emissivity, self.SB_const = args.h_conv, args.emissivity, args.SB_const

        self.L0, self.Qg, self.L_p, self.L_g = args.L0, args.Qg, args.L_p, args.L_g
        self.delta_fp, self.delta_fg = args.delta_fp, args.delta_fg
        self.sigma_p, self.sigma_g0 = args.sigma_p, args.sigma_g0
        self.gamma, self.degree_aniso = args.gamma, args.degree_aniso

        self.Lp_T = self.L0 * ti.exp(-self.Qg / (self.gas_const * self.T_liquid))
        self.m_p = 3 / 4 * self.sigma_p / (self.delta_fp * self.L_p)
        self.m_g = 3 / 4 * self.sigma_g0 / (self.delta_fg * self.L_g)
        self.kappa_p = 3 / 4 * self.sigma_p * self.L_p

        self.temp_cal = ti.Struct.field({
            'temp': float,
            'temp_td': float,
            'kappa': float,
            'rho_cp': float,
            'zeta': float,
            'len_grad_zeta': float,
            'q_con_rad_vap': float,

            'phi': float,
            'phi_td': float,
            'df_dphi': float,
            'Lg_T': float,
            'kappa_g': float,
            'ori_id': int,

            # 'phi0': float,
            'sum_phi2': float,
        }, shape=(self.Nx, self.Ny, self.Nz, self.num_oris))

        self.phi_xyz = ti.field(dtype=float, shape=(self.Nz, self.Ny, self.Nx, self.num_oris))
        self.edges = ti.field(dtype=float, shape=(self.Nz, self.Ny, self.Nx, self.num_oris, 3))
        self.directions = ti.field(dtype=float, shape=(self.Nx * self.Ny * self.Nz, self.num_oris, 3))

    @ti.func
    def grad_var(self, var, len_grad_var):

        for ix, iy, iz in ti.ndrange((1, self.Nx - 1), (1, self.Ny - 1), (1, self.Nz - 1)):
            grad_x = (var[ix + 1, iy, iz, 0] - var[ix - 1, iy, iz, 0]) / (2 * self.dxyz)
            grad_y = (var[ix, iy + 1, iz, 0] - var[ix, iy - 1, iz, 0]) / (2 * self.dxyz)
            grad_z = (var[ix, iy, iz + 1, 0] - var[ix, iy, iz - 1, 0]) / (2 * self.dxyz)

            len_grad_var[ix, iy, iz, 0] = tm.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)

    @ti.kernel
    def args_update(self, temp_cal: ti.template()):

        z0 = ti.max(0, args.Nz - args.reduced_Nz)
        for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):
            args.phi[ix, iy, z0 + iz, self.num_oris] = temp_cal[ix, iy, iz, 0].ori_id

        for ix, iy, iz, g in ti.ndrange(self.Nx, self.Ny, self.Nz, self.num_oris):
            args.phi[ix, iy, z0 + iz, g] = temp_cal[ix, iy, iz, g].phi

    @ti.kernel
    def set_bc(self, phi: ti.template()):

        for ix, iy, iz, g in ti.ndrange(self.Nx, self.Ny, self.Nz, self.num_oris):
            phi[0, iy, iz, g] = phi[1, iy, iz, g]
            phi[ix, 0, iz, g] = phi[ix, 1, iz, g]
            phi[ix, iy, 0, g] = phi[ix, iy, 1, g]

            phi[self.Nx - 1, iy, iz, g] = phi[self.Nx - 2, iy, iz, g]
            phi[ix, self.Ny - 1, iz, g] = phi[ix, self.Ny - 2, iz, g]
            phi[ix, iy, self.Nz - 1, g] = phi[ix, iy, self.Nz - 2, g]

    @ti.kernel
    def phi_update(self, temp_cal: ti.template()):

        for ix, iy, iz, g in ti.ndrange(self.Nx, self.Ny, self.Nz, self.num_oris):
            temp_cal[ix, iy, iz, g].phi_td = temp_cal[ix, iy, iz, g].phi

        for ix, iy, iz, g in ti.ndrange((1, self.Nx - 1), (1, self.Ny - 1), (1, self.Nz - 1), self.num_oris):

            temp, zeta0 = temp_cal[ix, iy, iz, 0].temp, temp_cal[ix, iy, iz, 0].zeta
            zeta_e, zeta_w = temp_cal[ix + 1, iy, iz, 0].zeta, temp_cal[ix - 1, iy, iz, 0].zeta
            zeta_n, zeta_s = temp_cal[ix, iy + 1, iz, 0].zeta, temp_cal[ix, iy - 1, iz, 0].zeta
            zeta_f, zeta_b = temp_cal[ix, iy, iz + 1, 0].zeta, temp_cal[ix, iy, iz - 1, 0].zeta

            a_e = -temp_cal[ix + 1, iy, iz, 0].Lg_T * (-temp_cal[ix + 1, iy, iz, g].kappa_g) * self.dxyz * zeta_e
            a_w = -temp_cal[ix - 1, iy, iz, 0].Lg_T * (-temp_cal[ix - 1, iy, iz, g].kappa_g) * self.dxyz * zeta_w
            a_n = -temp_cal[ix, iy + 1, iz, 0].Lg_T * (-temp_cal[ix, iy + 1, iz, g].kappa_g) * self.dxyz * zeta_n
            a_s = -temp_cal[ix, iy - 1, iz, 0].Lg_T * (-temp_cal[ix, iy - 1, iz, g].kappa_g) * self.dxyz * zeta_s
            a_f = -temp_cal[ix, iy, iz + 1, 0].Lg_T * (-temp_cal[ix, iy, iz + 1, g].kappa_g) * self.dxyz * zeta_f
            a_b = -temp_cal[ix, iy, iz - 1, 0].Lg_T * (-temp_cal[ix, iy, iz - 1, g].kappa_g) * self.dxyz * zeta_b
            ap0 = self.dxyz ** 3 / self.dt
            ap = ap0 + a_e + a_w + a_n + a_s + a_f + a_b

            rhs = (ap0 * temp_cal[ix, iy, iz, g].phi_td -
                   temp_cal[ix, iy, iz, 0].Lg_T * temp_cal[ix, iy, iz, g].df_dphi * self.dxyz ** 3 * zeta0 +
                   a_e * temp_cal[ix + 1, iy, iz, g].phi_td + a_w * temp_cal[ix - 1, iy, iz, g].phi_td +
                   a_n * temp_cal[ix, iy + 1, iz, g].phi_td + a_s * temp_cal[ix, iy - 1, iz, g].phi_td +
                   a_f * temp_cal[ix, iy, iz + 1, g].phi_td + a_b * temp_cal[ix, iy, iz - 1, g].phi_td)

            r = tm.clamp((self.T_liquid - temp) / (self.T_liquid - self.T_solid), 0., 1.)
            ratio = r ** 3 * (6 * r ** 2 - 15 * r + 10)
            region = (self.T_solid <= temp <= self.T_liquid)
            phi_noise = (5e-04 + 1e-04 * (ti.random(float) - 0.5)) * ratio ** 1 * region * zeta0

            _phi = tm.clamp(rhs / ap + phi_noise, 0., 1.)
            temp_cal[ix, iy, iz, g].phi = tm.min(_phi, zeta0 * ratio)

            if temp_cal[ix, iy, iz, g].phi >= 0.5:
                temp_cal[ix, iy, iz, 0].ori_id = g + 1

        ####-------------------
        temp_cal.sum_phi2.fill(0)
        for ix, iy, iz, g in ti.ndrange(self.Nx, self.Ny, self.Nz, self.num_oris):
            temp_cal[ix, iy, iz, 0].sum_phi2 += temp_cal[ix, iy, iz, g].phi ** 2

    @ti.kernel
    def df_cal(self, temp_cal: ti.template()):

        for ix, iy, iz, g in ti.ndrange(self.Nx, self.Ny, self.Nz, self.num_oris):

            temp, phi = temp_cal[ix, iy, iz, 0].temp, temp_cal[ix, iy, iz, g].phi
            delta_fg = tm.log(self.latent_s) * (temp - self.T_ambient) / self.T_ambient
            ag, bg, cg = -delta_fg, delta_fg, 2 * self.gamma * delta_fg
            _term = 0.
            for gg in range(self.num_oris):
                _term += phi * temp_cal[ix, iy, iz, gg].phi ** 2
            _term -= phi ** 3
            temp_cal[ix, iy, iz, g].df_dphi = self.m_g * (ag * phi + bg * phi ** 3 + cg * _term)

    @ti.kernel
    def anisotropy_update(self, phi: ti.template(), kappa_g: ti.template()):

        Nx, Ny, Nz, num_oris = self.Nx, self.Ny, self.Nz, self.num_oris

        for ix0, iy0, iz0, g0 in ti.ndrange(Nx, Ny, Nz, num_oris):
            index = ix0 * Ny * Nz * num_oris + iy0 * Nz * num_oris + iz0 * num_oris + g0
            iz = index // (Ny * Nx * num_oris)
            iy = (index - iz * Ny * Nx * num_oris) // (Nx * num_oris)
            ix = (index - iz * Ny * Nx * num_oris - iy * Nx * num_oris) // num_oris
            g = index - iz * Ny * Nx * num_oris - iy * Nx * num_oris - ix * num_oris

            self.phi_xyz[iz, iy, ix, g] = phi[ix0, iy0, iz0, g0]

        for ix, iy, iz, g in ti.ndrange(Nx, Ny, Nz, num_oris):
            self.edges[iz, iy, ix, g, 0] = (self.phi_xyz[iz, iy, ix + (ix < Nx - 1), g] -
                                            self.phi_xyz[iz, iy, ix - (ix > 0), g])
            self.edges[iz, iy, ix, g, 1] = (self.phi_xyz[iz, iy + (iy < Ny - 1), ix, g] -
                                            self.phi_xyz[iz, iy - (iy > 0), ix, g])
            self.edges[iz, iy, ix, g, 2] = (self.phi_xyz[iz + (iz < Nz - 1), iy, ix, g] -
                                            self.phi_xyz[iz - (iz > 0), iy, ix, g])

        for iz0, iy0, ix0, g, d in ti.ndrange(Nz, Ny, Nx, num_oris, 3):
            index = iz0 * Ny * Nx * num_oris * 3 + iy0 * Nx * num_oris * 3 + ix0 * num_oris * 3 + g * 3 + d
            ix = index // (num_oris * 3)
            iy = (index - ix * num_oris * 3) // 3
            iz = index - ix * num_oris * 3 - iy * 3

            self.directions[ix, iy, iz] = self.edges[iz0, iy0, ix0, g, d]

        for ix, iy, iz, g in ti.ndrange(Nx, Ny, Nz, num_oris):

            angles = math.pi
            i = ix * Ny * Nz + iy * Nz + iz

            for d in range(3):
                go_Vec = ti.Vector([args.grain_ori[d, g, 0], args.grain_ori[d, g, 1], args.grain_ori[d, g, 2]])
                ed_Vec = ti.Vector([self.directions[i, g, 0], self.directions[i, g, 1], self.directions[i, g, 2]])

                cosines, demo = tm.dot(go_Vec, ed_Vec), tm.length(ed_Vec)
                cosines0 = 0. if demo == 0 else cosines / demo
                tmp_angle0 = tm.acos(cosines0)
                tmp_angle0 = tmp_angle0 if tmp_angle0 < math.pi / 2 else math.pi - tmp_angle0
                angles = tmp_angle0 if tmp_angle0 < angles else angles

            aniso_term = 1. + args.degree_aniso * (tm.cos(angles) ** 4 + tm.sin(angles) ** 4)
            kappa_g[ix, iy, iz, g] = 3 / 4 * self.L_g * self.sigma_g0 * aniso_term

    @ti.kernel
    def crystal_init(self, temp_cal: ti.template()):

        z0 = ti.max(0, args.Nz - args.reduced_Nz)
        for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):
            i, temp = (ix, iy, iz), temp_cal[ix, iy, iz, 0].temp
            temp_cal[i, 0].Lg_T = (self.L0 * (temp / self.T_ambient) ** 0.7 *
                                   ti.exp(- self.Qg / (self.gas_const * temp))) * 1   ###20250305 #########

            for g in range(self.num_oris):
                temp_cal[ix, iy, iz, g].phi = args.phi[ix, iy, iz + z0, g]

    @ti.kernel
    def temp_fvm(self, temp_cal: ti.template()):

        for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):
            temp_cal[ix, iy, iz, 0].temp_td = temp_cal[ix, iy, iz, 0].temp

        self.grad_var(temp_cal.zeta, temp_cal.len_grad_zeta)

        for ix, iy, iz in ti.ndrange((1, self.Nx - 1), (1, self.Ny - 1), (1, self.Nz - 1)):

            a_e, a_w = temp_cal[ix + 1, iy, iz, 0].kappa * self.dxyz, temp_cal[ix - 1, iy, iz, 0].kappa * self.dxyz
            a_n, a_s = temp_cal[ix, iy + 1, iz, 0].kappa * self.dxyz, temp_cal[ix, iy - 1, iz, 0].kappa * self.dxyz
            a_f, a_b = temp_cal[ix, iy, iz + 1, 0].kappa * self.dxyz, temp_cal[ix, iy, iz - 1, 0].kappa * self.dxyz
            ap0 = temp_cal[ix, iy, iz, 0].rho_cp * self.dxyz ** 3 / self.dt
            ap = ap0 + a_e + a_w + a_n + a_s + a_f + a_b

            temp_rhs_2 = -(temp_cal[ix, iy, iz, 0].q_con_rad_vap * temp_cal[ix, iy, iz, 0].len_grad_zeta *
                           temp_cal[ix, iy, iz, 0].rho_cp / self.avg_rho_cp

            b = ap0 * temp_cal[ix, iy, iz, 0].temp_td + temp_rhs_2 * self.dxyz ** 3

            rhs = (a_e * temp_cal[ix + 1, iy, iz, 0].temp_td + a_w * temp_cal[ix - 1, iy, iz, 0].temp_td +
                   a_n * temp_cal[ix, iy + 1, iz, 0].temp_td + a_s * temp_cal[ix, iy - 1, iz, 0].temp_td +
                   a_f * temp_cal[ix, iy, iz + 1, 0].temp_td + a_b * temp_cal[ix, iy, iz - 1, 0].temp_td + b)

            temp_cal[ix, iy, iz, 0].temp = tm.clamp(rhs / ap, self.T_ambient, self.T_vapor)

        for ix, iy, ex in ti.ndrange(self.Nx, self.Ny, 2):
            temp_cal[ix, iy, ex, 0].temp = args.T_subplate
            temp_cal[ix, iy, self.Nz - 1, 0].temp = temp_cal[ix, iy, self.Nz - 2, 0].temp

        z0 = ti.max(0, args.Nz - args.reduced_Nz)
        for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):
            args.temp[ix, iy, iz + z0] = temp_cal[ix, iy, iz, 0].temp

    @ti.kernel
    def temp_terms(self, temp_cal: ti.template()):

        for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):
            i = (ix, iy, iz, 0)

            zeta, temp = temp_cal[i].zeta, temp_cal[i].temp
            Pv_T = self.p0 * ti.exp(self.latent_v / self.gas_const * (1 / self.T_vapor - 1 / temp))
            q_con = self.h_conv * (temp - self.T_ambient)
            q_rad = self.SB_const * self.emissivity * (temp ** 4 - self.T_ambient ** 4)
            q_vap = zeta * Pv_T * ti.sqrt(self.mo / (2 * math.pi * self.Kb * temp)) * self.latent_v

            temp_cal[i].q_con_rad_vap = q_con + q_rad + q_vap

    @ti.kernel
    def temp_fieldassign(self, temp_cal: ti.template()):

        z0 = ti.max(0, args.Nz - args.reduced_Nz)
        for ix, iy, iz in ti.ndrange(self.Nx, self.Ny, self.Nz):
            temp_cal[ix, iy, iz, 0].temp = args.temp[ix, iy, iz + z0]
            temp_cal[ix, iy, iz, 0].zeta = args.zeta[ix, iy, iz + z0]

            i = (ix, iy, iz, 0)
            
            zeta, temp = temp_cal[i].zeta, temp_cal[i].temp
            rho = zeta * self.rho_metal + (1 - zeta) * self.rho_gas
            
            r = tm.clamp((temp - self.T_solid) / (self.T_liquid - self.T_solid), 0., 1.)
            f_liquid = zeta * r ** 3 * (6 * r ** 2 - 15 * r + 10)
            
            temp_cal[i].rho_cp = rho * ((zeta - f_liquid) * self.cp_solid +
                                         f_liquid * self.cp_liquid + (1 - zeta) * self.cp_gas)
            
            temp_cal[i].kappa = ((zeta - f_liquid) * self.kappa_solid +
                                  f_liquid * self.kappa_liquid + (1 - zeta) * self.kappa_gas)

    def temp_coolingdown(self):

        self.temp_fieldassign(self.temp_cal)
        self.temp_terms(self.temp_cal)
        self.temp_fvm(self.temp_cal)

        self.crystal_init(self.temp_cal)
        self.anisotropy_update(self.temp_cal.phi, self.temp_cal.kappa_g)
        self.df_cal(self.temp_cal)
        self.phi_update(self.temp_cal)
        self.set_bc(self.temp_cal.phi)
        self.args_update(self.temp_cal)

        args.sum_phi2 = self.temp_cal.sum_phi2.to_numpy()  ###########


os.makedirs('result', exist_ok=True)  # Make dir for output

def integrator():

    t0 = time.time()

    laser_track(), get_unique_ori()
    pp_fields = post_process()
    dem_solver, cfd_solver, pfm_solver = PowderDEM(), CfdFunc(), PfmFunc()

    dem_solver.powder_dem()

    SimulationSteps, _ = args.laser_info.shape

    for istep in range(SimulationSteps):

        cfd_solver.temp_solver(istep)
        if args.iflag_cfd == 1:
            print('cfd', istep)
            break

        pfm_solver.AC_solver()
        if args.iflag_pfm == 1:
            print('pfm', istep)
            break

        args.time += args.dt_CFD * (istep >= 1)

        if int(args.laser_info[istep, 3]):

            cooling_solver = track_coolingdown()
            for iistep in range(4000):
                cooling_solver.temp_coolingdown()

                args.time += args.dt_CFD

            pp_fields.output()

        if int(args.laser_info[istep, 4]):
            args.layer_switch = 1
            args.layer_order += 1

            pp_fields.update()
            if istep < SimulationSteps - 1:
                dem_solver = PowderDEM()
                dem_solver.powder_dem()

            cfd_solver, pfm_solver = CfdFunc(), PfmFunc()
            temp00 = np.ones([args.Nx, args.Ny, args.total_Nz], dtype=np.float32) * args.T_ambient
            args.temp = ti.field(dtype=float, shape=(args.Nx, args.Ny, args.total_Nz))
            args.temp.from_numpy(temp00)

    pp_fields.output()
    t1 = time.time()
    print(f'duration: {(t1 - t0)/ 3600} h')


if __name__ == "__main__":
    integrator()
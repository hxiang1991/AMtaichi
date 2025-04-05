import numpy as np
import taichi as ti
import taichi.math as tm
import scipy.io as scio
from arguments import args
from pyevtk.hl import gridToVTK
from orix.quaternion import Orientation
from scipy.spatial.transform import Rotation as R

ti.init(arch=ti.gpu, default_fp=ti.f32, device_memory_fraction=0.95)
vec3_f = ti.types.vector(3, ti.f32)

ti.sync()
ti.init(kernel_profiler=True)


def laser_track():

    laser_vel, dt_laser, num_layer = args.laser_vel, args.dt_CFD, args.num_layer

    filename_1 = 'ScanStrategy/scan_strategy'
    filename_2 = "%d" % args.scan_strategy

    scanpath_list = []
    with open(f'{filename_1}_{filename_2}.txt', 'r') as f:
        for line in f:
            startpos_x = float(line.strip().split('\t')[0]) / 1e03   # convert the mm into m
            startpos_y = float(line.strip().split('\t')[1]) / 1e03
            endpos_x = float(line.strip().split('\t')[2]) / 1e03
            endpos_y = float(line.strip().split('\t')[3]) / 1e03
            ang = float(line.strip().split('\t')[4]) * np.pi/180
            layer_switch = float(line.strip().split('\t')[5])
            scanpath_list.append([startpos_x, startpos_y, endpos_x, endpos_y, ang, layer_switch])
    scanpath = np.array(scanpath_list)

    ts, xs, ys, line_sw, layer_sw = [], [], [], [], []
    t_pre = 0.0
    for _, scanpos in enumerate(scanpath):
        start_pos = np.array([scanpos[0], scanpos[1]])
        end_pos = np.array([scanpos[2], scanpos[3]])
        traveled_dist = np.linalg.norm(end_pos - start_pos)
        traveled_time = traveled_dist / laser_vel
        move = round(traveled_time / dt_laser)

        ts_seg = t_pre + np.arange(move + 1) * dt_laser
        xs_seg = np.linspace(scanpos[0], scanpos[2], len(ts_seg))
        ys_seg = np.linspace(scanpos[1], scanpos[3], len(ts_seg))
        ls_seg = np.linspace(0, 0, len(ts_seg))
        ls_seg[-1] = 1
        sw_seg = np.linspace(scanpos[5], scanpos[5], len(ts_seg), dtype=int)
        sw_seg[: -1] = 0
        sw_seg = np.array([bool(i) for i in sw_seg])

        ts.append(ts_seg)
        xs.append(xs_seg)
        ys.append(ys_seg)
        line_sw.append(ls_seg)
        layer_sw.append(sw_seg)

        t_pre = t_pre + traveled_time + dt_laser

    line_sw, layer_sw = np.hstack(line_sw), np.hstack(layer_sw)
    ts, xs, ys = np.hstack(ts), np.hstack(xs, dtype=np.float32), np.hstack(ys, dtype=np.float32)

    laser_info = np.stack((ts, xs, ys, line_sw, layer_sw), axis=1).astype(np.float32)
    args.laser_info = ti.field(dtype=ti.f32, shape=(len(ts), 5))
    args.laser_info.from_numpy(laser_info)
    args.laser_step = len(ts)


def get_unique_ori():

    num_oris = args.num_oris

    ori2 = Orientation.random(num_oris)
    dx, dy, dz = np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])
    scipy_quat = np.concatenate((ori2.data[:, 1:], ori2.data[:, :1]), axis=1)
    r = R.from_quat(scipy_quat)
    grain_directions = np.stack((r.apply(dx), r.apply(dy), r.apply(dz))).astype(np.float32)

    args.grain_ori = ti.field(dtype=ti.f32, shape=(3, num_oris, 3))
    args.grain_ori.from_numpy(grain_directions)


@ti.data_oriented
class post_process:

    def __init__(self):

        self.num_oris = args.num_oris
        args.dxyz = args.domain_length / args.Nx

        self.dt, self.dxyz = args.dt_PFM, args.dxyz

        args.layer_Nz = int(args.layer_thick/self.dxyz)
        args.reduced_Nz = 2 * args.layer_Nz   ###

        self.Nx, self.Ny = args.Nx, args.Ny
        self.Nz = 2 + ti.min(args.Nz_old, 2 * args.layer_Nz)

        args.layer_switch = 0
        args.temp0 = ti.field(dtype=float, shape=(self.Nx, self.Ny, args.reduced_Nz))  ###

    @ti.kernel
    def field_update(self, temp: ti.template(), temp0: ti.template(), zeta: ti.template()):

        for i in ti.grouped(temp0):
            temp0[i] = args.T_subplate
        x0, x1 = 10, args.Nx - 10
        y0, y1 = 10, args.Ny - 10
        args.z_height[0] = args.Nz_old
        for ix, iy in ti.ndrange((x0, x1), (y0, y1)):
            for iz in range(args.Nz_old):
                if zeta[ix, iy, args.Nz_old - iz] >= 0.5 > zeta[ix, iy, args.Nz_old - iz + 1]:
                    args.z_height[0] = tm.min(args.z_height[0], args.Nz_old - iz)
                    break

    @ti.kernel
    def update_nz(self, zeta: ti.template()) -> int:

        tmp_Nz = 0
        x0, x1 = 10, self.Nx - 10
        y0, y1 = 10, self.Ny - 10
        for ix, iy in ti.ndrange((x0, x1), (y0, y1)):
            for iz in range(args.total_Nz):
                if zeta[ix, iy, args.total_Nz - iz - 1] > 0. == zeta[ix, iy, args.total_Nz - iz]:
                    tmp_Nz = tm.max(tmp_Nz, args.total_Nz - iz)
                    break

        return tmp_Nz

    def update(self):

        args.Nz_old = self.update_nz(args.zeta)
        args.temp0 = ti.field(dtype=float, shape=(self.Nx, self.Ny, args.Nz_old))
        self.field_update(args.temp, args.temp0, args.zeta)

    def output(self):

        xcor = np.linspace(0, args.domain_length, args.Nx).astype(np.float32)
        ycor = np.linspace(0, args.domain_width, args.Ny).astype(np.float32)
        zcor = np.linspace(0, args.total_height, args.total_Nz).astype(np.float32)

        temp_np = args.temp.to_numpy()
        zeta, phi0 = args.zeta.to_numpy(), args.phi.to_numpy()

        temp = temp_np
        ori_id = phi0[:, :, :, -1]

        print(f'[Taichi] Exporting {args.time * 1e03: 06f}-ms result...')
        gridToVTK(f'./result/{args.time * 1e03: 06f}-ms', xcor, ycor, zcor,
                  pointData={
                             'temp': np.ascontiguousarray(temp),
                             'zeta': np.ascontiguousarray(zeta),
                             'ori_id': np.ascontiguousarray(ori_id),
                             })


class mesh3d():

    def __init__(self, domain, N):

        self.shape = N
        self.dim = len(N)
        self.generate_mesh(domain, N)
        self.get_neighbor(N)
        self.get_surface(N)
        
    def generate_mesh(self, domain, N):

        Nx, Ny, Nz = N
        domain_x, domain_y, domain_z = domain
        self.cell_num = Nx * Ny * Nz
        x = np.linspace(0., domain_x, Nx + 1)
        y = np.linspace(0., domain_y, Ny + 1)
        z = np.linspace(0., domain_z, Nz + 1)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

        xc = np.array((xx[1:, 1:, 1:] + xx[:-1, 1:, 1:]) / 2)
        yc = np.array((yy[1:, 1:, 1:] + yy[1:, :-1, 1:]) / 2)
        zc = np.array((zz[1:, 1:, 1:] + zz[1:, 1:, :-1]) / 2)

        self.dX = np.array([domain_x / Nx, domain_y / Ny, domain_z / Nz])
        self.Xc = np.stack((xc, yc, zc), axis=3).reshape(-1, 3)

        self.dV = self.dX[0] * self.dX[1] * self.dX[2]
        self.D = np.array([-self.dX[0], self.dX[0], -self.dX[1], self.dX[1], -self.dX[2],self.dX[2]])
        self.dS = np.array([
            -self.dX[1] * self.dX[2], self.dX[1] * self.dX[2],
            -self.dX[2] * self.dX[0], self.dX[2] * self.dX[0],
            -self.dX[0] * self.dX[1], self.dX[0] * self.dX[1]])

    def get_neighbor(self, N):

        Nx, Ny, Nz = N
        cell_idx = np.arange(0, Nx * Ny * Nz).reshape((Nx, Ny, Nz))
        cell_index = np.pad(cell_idx, 1, "symmetric")
        self.cell_conn = np.stack(
            (cell_index[1:-1, 1:-1, 1:-1], cell_index[:-2, 1:-1, 1:-1],
             cell_index[2:, 1:-1, 1:-1], cell_index[1:-1, :-2, 1:-1],
             cell_index[1:-1, 2:, 1:-1], cell_index[1:-1, 1:-1, :-2],
             cell_index[1:-1, 1:-1, 2:]), axis=3).reshape(-1, 7)

    def get_surface(self, N):

        Nx, Ny, Nz = N
        cell_idx = np.arange(0, Nx * Ny * Nz).reshape((Nx, Ny, Nz))
        x_neg_idx = cell_idx[0, :, :].flatten()
        x_pos_idx = cell_idx[-1, :, :].flatten()
        y_neg_idx = cell_idx[:, 0, :].flatten()
        y_pos_idx = cell_idx[:, -1, :].flatten()
        z_neg_idx = cell_idx[:, :, 0].flatten()
        z_pos_idx = cell_idx[:, :, -1].flatten()
        self.surf_cell = np.concatenate((x_neg_idx, x_pos_idx, y_neg_idx, y_pos_idx, z_neg_idx, z_pos_idx))

        self.surf_sets = [
            np.arange(0, Ny * Nz),
            np.arange(Ny * Nz, Ny * Nz * 2),
            np.arange(Ny * Nz * 2, Ny * Nz * 2 + Nx * Nz),
            np.arange(Ny * Nz * 2 + Nx * Nz, Ny * Nz * 2 + Nx * Nz * 2),
            np.arange(Ny * Nz * 2 + Nx * Nz * 2, Ny * Nz * 2 + Nx * Nz * 2 + Nx * Ny),
            np.arange(Ny * Nz * 2 + Nx * Nz * 2 + Nx * Ny, Ny * Nz * 2 + Nx * Nz * 2 + Nx * Ny * 2)]

        self.surf_set_num = [Ny * Nz, Ny * Nz, Nx * Nz, Nx * Nz, Ny * Nx, Ny * Nx]

        self.surf_orient = np.concatenate(
            (np.ones(Ny * Nz) * 0, np.ones(Ny * Nz) * 1, np.ones(Nx * Nz) * 2,
             np.ones(Nx * Nz) * 3, np.ones(Ny * Nx) * 4, np.ones(Ny * Nx) * 5))

        surf_x_neg = self.Xc[x_neg_idx] - np.array([self.dX[0] / 2., 0., 0.])
        surf_x_pos = self.Xc[x_pos_idx] + np.array([self.dX[0] / 2., 0., 0.])
        surf_y_neg = self.Xc[y_neg_idx] - np.array([0., self.dX[1] / 2., 0.])
        surf_y_pos = self.Xc[y_pos_idx] + np.array([0., self.dX[1] / 2., 0.])
        surf_z_neg = self.Xc[z_neg_idx] - np.array([0., 0., self.dX[2] / 2.])
        surf_z_pos = self.Xc[z_pos_idx] + np.array([0., 0., self.dX[2] / 2.])
        self.surface = np.concatenate((surf_x_neg, surf_x_pos, surf_y_neg,surf_y_pos, surf_z_neg, surf_z_pos))
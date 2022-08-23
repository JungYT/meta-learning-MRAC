import fym
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scipy.linalg
from fym.core import BaseEnv, BaseSystem
from fym.utils.linearization import jacob_analytic
from fym.utils.rot import (angle2dcm, angle2quat, dcm2angle, hat, omega2deuler,
                           quat2angle, quat2dcm)
from numpy.linalg import norm
from plotly.subplots import make_subplots as subplot


def cross(x, y):
    return np.cross(x, y, axis=0)


class Multicopter(BaseEnv):
    g = 9.81
    m = 1.00
    r = 0.24
    J = np.diag([8.1, 8.1, 14.2]) * 1e-3
    Jinv = np.linalg.inv(J)
    b = 5.42e-5
    d = 1.1e-6
    Kf = np.diag([5.567, 5.567, 6.354]) * 1e-4
    Kt = np.diag([5.567, 5.567, 6.354]) * 1e-4
    rotorf_min = 0
    rotorf_max = 20
    e3 = np.vstack((0, 0, 1))
    nrotors = 4
    L = np.array(
        [
            [1, 1, 1, 1],
            [0, -r, 0, r],
            [r, 0, -r, 0],
            [-d / b, d / b, -d / b, d / b],
        ]
    )
    Lambda = np.eye(4)

    def __init__(
        self,
        pos=np.zeros((3, 1)),
        vel=np.zeros((3, 1)),
        quat=angle2quat(0, 0, 0),
        omega=np.zeros((3, 1)),
    ):
        super().__init__()
        self.pos = BaseSystem(pos)
        self.vel = BaseSystem(vel)
        self.quat = BaseSystem(quat)
        self.omega = BaseSystem(omega)

    def deriv(self, pos, vel, quat, omega, rotorfs):
        u = self.L @ rotorfs
        fT, M = u[:1], u[1:]
        C_bi = quat2dcm(quat)
        wx = omega[0].item()
        wy = omega[1].item()
        wz = omega[2].item()

        dpos = vel
        dvel = (
            self.m * self.g * self.e3 + C_bi.T @ (-fT * self.e3) - self.Kf @ vel
        ) / self.m
        dquat = (
            0.5
            * np.array(
                [
                    [0, -wx, -wy, -wz],
                    [wx, 0, wz, -wy],
                    [wy, -wz, 0, wx],
                    [wz, wy, -wx, 0],
                ]
            )
            @ quat
        )
        domega = self.Jinv @ (
            M - cross(omega, self.J @ omega) - norm(omega) * self.Kt @ omega
        )
        return dpos, dvel, dquat, domega

    def set_dot(self, t, rotorfs_cmd):
        pos, vel, quat, omega = self.observe_list()
        rotorfs = self.saturate(t, rotorfs_cmd)
        dots = self.deriv(pos, vel, quat, omega, rotorfs)
        self.pos.dot, self.vel.dot, self.quat.dot, self.omega.dot = dots
        return dict(rotorfs=rotorfs)

    def saturate(self, t, rotorfs_cmd):
        rotorfs = np.clip(rotorfs_cmd, self.rotorf_min, self.rotorf_max)
        return self.Lambda @ rotorfs


def wrap(func):
    def deriv(x, u):
        pos, vel, euler, omega = x[:3], x[3:6], x[6:9], x[9:]
        phi, theta, psi = euler.ravel()
        quat = angle2quat(psi, theta, phi)
        dpos, dvel, _, domega = func(pos, vel, quat, omega, u)
        deuler = omega2deuler(omega, phi, theta)
        xdot = np.vstack((dpos, dvel, deuler, domega))
        return xdot

    return deriv


class MRAC(BaseEnv):
    def __init__(self, Theta, A_ref, B_ref, x_ref, P_ref, B, Gamma, C):
        super().__init__()
        self.e_yI = BaseSystem(np.zeros((4, 1)))
        self.Theta = BaseSystem(Theta)
        self.Ref = RefModel(A_ref, B_ref, x_ref)
        self.P_ref = P_ref
        self.B = B
        self.Gamma = Gamma
        self.C = C

    def set_dot(self, t, x_p, r):
        e_yI = self.e_yI.state
        x = np.vstack((e_yI, x_p))
        y = self.C @ x
        x_ref = self.Ref.x_ref.state
        e = x - x_ref
        Phi = self.get_basis(x, x_p)
        self.Theta.dot = self.Gamma @ Phi @ e.T @ self.P_ref @ self.B
        self.Ref.set_dot(t, r)
        self.e_yI.dot = y - r
        return Phi, e

    def get_action(self, x_p):
        e_yI = self.e_yI.state
        x = np.vstack((e_yI, x_p))
        Phi = self.get_basis(x, x_p)
        Theta = self.Theta.state
        action = -Theta.T @ Phi
        return action

    def get_basis(self, x, x_p):
        Phi_d = x_p**2
        Phi = np.vstack((-x, Phi_d))
        return Phi


class RefModel(BaseEnv):
    def __init__(self, A_ref, B_ref, x_ref):
        super().__init__()
        self.x_ref = BaseSystem(x_ref)
        self.A_ref = A_ref
        self.B_ref = B_ref

    def set_dot(self, t, r):
        x_ref = self.x_ref.state
        self.x_ref.dot = self.A_ref @ x_ref + self.B_ref @ r


class Env(BaseEnv):
    def __init__(self, env_config):
        super().__init__(**env_config["sim"])
        self.plant = Multicopter(**env_config["plant"])
        self.mrac = MRAC(**env_config["mrac"])

    def step(self, action):
        *_, done = self.update(action=action)
        next_obs = self.observe()
        reward = self.get_reward()
        info = {}
        return next_obs, reward, done, info

    def set_dot(self, t, action):
        r = action.reshape(-1, 1)
        x_p = self.observe().reshape(-1, 1)
        Phi, e = self.mrac.set_dot(t, x_p, r)
        rotorfs_cmd = self.mrac.get_action(x_p)
        rotorfs = self.plant.set_dot(t, rotorfs_cmd)

        return dict(
            t=t,
            **self.plant.observe_dict(),
            **self.mrac.observe_dict(),
            **rotorfs,
            rotorfs_cmd=rotorfs_cmd,
            obs=x_p,
            r=r,
            Phi=Phi,
            e=e,
        )

    def observe(self):
        pos, vel, quat, omega = self.plant.observe_list()
        yaw, pitch, roll = quat2angle(quat)
        obs = np.hstack((pos.ravel(), vel.ravel(), roll, pitch, yaw, omega.ravel()))
        return np.float32(obs)

    def get_reward(self):
        return 1

    def reset(self, random=True):
        super().reset()
        return self.observe()


def set_config():
    plant = Multicopter()
    pos0 = np.zeros((3, 1))
    vel0 = np.zeros((3, 1))
    euler0 = np.zeros((3, 1))
    quat0 = angle2quat(0, 0, 0)
    omega0 = np.zeros((3, 1))
    u0 = plant.m * plant.g * np.ones((4, 1)) / 4
    x0 = np.vstack((pos0, vel0, euler0, omega0))
    e_yI0 = np.zeros((4, 1))
    x_ref0 = np.vstack((e_yI0, x0))
    nxp = 12
    ny = 4
    nx = nxp + ny
    nu = 4
    A_jacob = jacob_analytic(wrap(plant.deriv), i=0)
    B_jacob = jacob_analytic(wrap(plant.deriv), i=1)
    A_p = A_jacob(x0, u0).squeeze()
    B_p = B_jacob(x0, u0).squeeze()
    C_p = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        ]
    )
    A = np.concatenate(
        (
            np.concatenate((np.zeros((ny, ny)), C_p), axis=1),
            np.concatenate((np.zeros((nxp, ny)), A_p), axis=1),
        ),
        axis=0,
    )
    B = np.concatenate((np.zeros((ny, nu)), B_p), axis=0)
    B_c = np.concatenate((-np.eye(ny), np.zeros((nxp, nu))), axis=0)
    C = np.concatenate((np.zeros((ny, ny)), C_p), axis=1)

    # Desing parameters
    Gamma = 1000 * np.eye(nx + nxp)
    Q_lqr = 10 * np.eye(nx)
    R_lqr = np.eye(nu)

    K_lqr, P_lqr = fym.clqr(A, B, Q_lqr, R_lqr)
    A_ref = A - B @ K_lqr
    # Q_ref = np.eye(nx)
    # P_ref = scipy.linalg.solve_continuous_lyapunov(A_ref.T, Q_ref)
    P_ref = P_lqr

    CONFIG = {
        "env": "quadcopter",
        "env_config": {
            "sim": {
                "dt": 0.001,
                "max_t": 10.0,
            },
            "plant": {
                "pos": pos0,
                "vel": vel0,
                "quat": quat0,
                "omega": omega0,
            },
            "mrac": {
                "Theta": np.zeros((nx + nxp, nu)),
                "A_ref": A_ref,
                "B_ref": B_c,
                "x_ref": x_ref0,
                "P_ref": P_ref,
                "B": B,
                "Gamma": Gamma,
                "C": C,
            },
        },
    }
    return CONFIG


def single_run(CONFIG):
    env = Env(CONFIG["env_config"])
    obs = env.reset()
    env.logger = fym.Logger("data.h5")
    env.logger.set_info(cfg=CONFIG)
    while True:
        y_cmd = np.array([1, 1, 1, 0])
        obs, reward, done, info = env.step(y_cmd)
        if done:
            break
    env.close()


def plot():
    data, cfg = fym.logging.load("./data.h5", with_info=True)
    t = data["t"].squeeze()
    pos = data["pos"].squeeze()
    vel = data["vel"].squeeze()
    quat = data["quat"].squeeze()
    omega = data["omega"].squeeze()
    rotorfs = data["rotorfs"].squeeze()
    rotorfs_cmd = data["rotorfs_cmd"].squeeze()
    r = data["r"].squeeze()
    obs = data["obs"].squeeze()
    euler = obs[:, 6:9] * 180 / np.pi
    pos_cmd = r[:, :3]
    yaw_cmd = r[:, 3] * 180 / np.pi
    theta = data["Theta"]
    Phi = data["Phi"].squeeze()
    e = data["e"].squeeze()
    x_ref = data['Ref']['x_ref'].squeeze()
    e_yI_ref = x_ref[:, :4]
    pos_ref = x_ref[:, 4:7]
    vel_ref = x_ref[:, 7:10]
    euler_ref = x_ref[:, 10:13]
    omega_ref = x_ref[:, 13:]

    tex_fonts = {
        "text.usetex": True,
        "font.family": "Times New Roman",
        "axes.grid": True,
    }
    plt.rcParams.update(tex_fonts)
    true_style = 'solid'
    ref_style = 'dashed'
    cmd_style = 'dotted'
    true_color = 'r'
    ref_color = 'b'
    cmd_color = 'k'

    #Trajectory
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection":"3d"})
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label='true', color=true_color, linestyle=true_style)
    ax.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2], label='ref.', color=ref_color, linestyle=ref_style)
    ax.plot(pos_cmd[:, 0], pos_cmd[:, 1], pos_cmd[:, 2], label='cmd.', color=cmd_color, linestyle=cmd_style)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Trajectory')
    plt.legend()
    fig.savefig('./traj.png', bbox_inches='tight')
    plt.close('all')

    # Position
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(t, pos[:, 0], color=true_color, label='true', linestyle=true_style)
    ax[0].plot(t, pos_ref[:, 0], color=ref_color, label='ref.', linestyle=ref_style)
    ax[0].plot(t, pos_cmd[:, 0], color=cmd_color, label='cmd.', linestyle=cmd_style)
    ax[0].set_ylabel('X [m]')
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(t, pos[:, 1], color=true_color, linestyle=true_style)
    ax[1].plot(t, pos_ref[:, 1], color=ref_color, linestyle=ref_style)
    ax[1].plot(t, pos_cmd[:, 1], color=cmd_color, linestyle=cmd_style)
    ax[1].set_ylabel('Y [m]')
    ax[1].axes.xaxis.set_ticklabels([])
    ax[2].plot(t, pos[:, 2], color=true_color, linestyle=true_style)
    ax[2].plot(t, pos_ref[:, 2], color=ref_color, linestyle=ref_style)
    ax[2].plot(t, pos_cmd[:, 2], color=cmd_color, linestyle=cmd_style)
    ax[2].set_ylabel('Z [m]')
    ax[0].legend()
    fig.suptitle('Position')
    fig.supxlabel('Time [s]')
    fig.align_ylabels(ax)
    fig.tight_layout()
    fig.savefig('./pos.png')
    plt.close('all')

    # Velocity
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(t, vel[:, 0], color=true_color, linestyle=true_style, label='true')
    ax[0].plot(t, vel_ref[:, 0], color=ref_color, linestyle=ref_style, label='ref.')
    ax[0].set_ylabel(r'$V_x$ [m/s]')
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(t, vel[:, 1], color=true_color, linestyle=true_style)
    ax[1].plot(t, vel_ref[:, 1], color=ref_color, linestyle=ref_style)
    ax[1].set_ylabel(r'$V_y$ [m/s]')
    ax[1].axes.xaxis.set_ticklabels([])
    ax[2].plot(t, vel[:, 2], color=true_color, linestyle=true_style)
    ax[2].plot(t, vel_ref[:, 2], color=ref_color, linestyle=ref_style)
    ax[2].set_ylabel(r'$V_z$ [m/s]')
    ax[0].legend()
    fig.suptitle('Velocity')
    fig.supxlabel('Time [s]')
    fig.align_ylabels(ax)
    fig.tight_layout()
    fig.savefig('./vel.png')
    plt.close('all')

    # Rotor forces
    fig, ax = plt.subplots(4, 1)
    ax[0].plot(t, rotorfs[:, 0], color=true_color, linestyle=true_style, label='true')
    ax[0].plot(t, rotorfs_cmd[:, 0], color=cmd_color, linestyle=cmd_style, label='cmd.')
    ax[0].set_ylabel(r'$F_1$ [N]')
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(t, rotorfs[:, 1], color=true_color, linestyle=true_style)
    ax[1].plot(t, rotorfs_cmd[:, 1], color=cmd_color, linestyle=cmd_style)
    ax[1].set_ylabel(r'$F_2$ [N]')
    ax[1].axes.xaxis.set_ticklabels([])
    ax[2].plot(t, rotorfs[:, 2], color=true_color, linestyle=true_style)
    ax[2].plot(t, rotorfs_cmd[:, 2], color=cmd_color, linestyle=cmd_style)
    ax[2].set_ylabel(r'$F_3$ [N]')
    ax[2].axes.xaxis.set_ticklabels([])
    ax[3].plot(t, rotorfs[:, 3], color=true_color, linestyle=true_style)
    ax[3].plot(t, rotorfs_cmd[:, 3], color=cmd_color, linestyle=cmd_style)
    ax[3].set_ylabel(r'$F_4$ [N]')
    ax[0].legend()
    fig.suptitle('Rotor force')
    fig.supxlabel('Time [s]')
    fig.align_ylabels(ax)
    fig.tight_layout()
    fig.savefig('./rotorfs.png')
    plt.close('all')

    # Angular velocity
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(t, omega[:, 0], color=true_color, linestyle=true_style, label='true')
    ax[0].plot(t, omega_ref[:, 0], color=ref_color, linestyle=ref_style, label='ref.')
    ax[0].set_ylabel(r'$\omega$ [rad/s]')
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(t, omega[:, 1], color=true_color, linestyle=true_style)
    ax[1].plot(t, omega_ref[:, 1], color=ref_color, linestyle=ref_style)
    ax[1].set_ylabel(r'$\omega$ [rad/s]')
    ax[1].axes.xaxis.set_ticklabels([])
    ax[2].plot(t, omega[:, 2], color=true_color, linestyle=true_style)
    ax[2].plot(t, omega_ref[:, 2], color=ref_color, linestyle=ref_style)
    ax[2].set_ylabel(r'$\omega$ [rad/s]')
    ax[0].legend()
    fig.suptitle('Angular velocity')
    fig.supxlabel('Time [s]')
    fig.align_ylabels(ax)
    fig.tight_layout()
    fig.savefig('./omega.png')
    plt.close('all')

    # Euler angle
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(t, euler[:, 0], color=true_color, linestyle=true_style)
    ax[0].plot(t, euler_ref[:, 0], color=ref_color, linestyle=ref_style)
    ax[0].set_ylabel(r'$\phi$ [deg]')
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(t, euler[:, 1], color=true_color, linestyle=true_style)
    ax[1].plot(t, euler_ref[:, 1], color=ref_color, linestyle=ref_style)
    ax[1].set_ylabel(r'$\theta$ [deg]')
    ax[1].axes.xaxis.set_ticklabels([])
    ax[2].plot(t, euler[:, 2], color=true_color, linestyle=true_style, label='true')
    ax[2].plot(t, euler_ref[:, 2], color=ref_color, linestyle=ref_style, label='ref.')
    ax[2].plot(t, yaw_cmd, color=cmd_color, linestyle=cmd_style, label='cmd')
    ax[2].set_ylabel(r'$\psi$ [deg]')
    ax[2].legend()
    fig.suptitle('Euler angle')
    fig.supxlabel('Time [s]')
    fig.align_ylabels(ax)
    fig.tight_layout()
    fig.savefig('./euler.png')
    plt.close('all')


if __name__ == "__main__":
    # CONFIG = set_config()
    # single_run(CONFIG)
    plot()

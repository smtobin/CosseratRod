import utils
import numpy as np


def main():
    # R = np.identity(3)
    # omega = utils.MatLog_SO3(R)
    # print(f"R: {R}, omega: {omega}")

    omega_in = np.array([1, 1, 0])
    omega_in = omega_in / np.linalg.norm(omega_in)
    theta_in = 0.00001
    R = utils.MatExp_so3(omega_in, theta_in)
    omega_out = utils.MatLog_SO3(R)
    print(f"Omega in: {omega_in*theta_in}, Omega out: {omega_out}")

if __name__ == "__main__":
    main()
__author__ = 'antonio franco'

'''
Copyright (C) 2019  Antonio Franco (antonio_franco@live.it)
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import mpmath as mp
from types import LambdaType
import matplotlib.pyplot as plt
import warnings


class NevilleDiff(object):
    """
    NevilleDiff: class based on J. N. Lyness and C. B. Moler, “Van der monde systems and numerical differentiation,”
     Numerische Mathematik, vol. 8, pp. 458–464, Aug 1966.
    """

    def __init__(self):
        super().__init__()
        self.C = []  # Coefficients of the interpolating polynomial
        self.r = mp.mpf(1)  # Output flag. 1: success, 0: failed to reach precision

    def __update__(self, k: int, p: int, x: list, fxk: float) -> None:
        """
        updates the coefficients of the interpolating polynomial C
        :param k: degree of the polynomial
        :param p: maximum derivative to calculate
        :param x: value of the point where to calculate the coefficients
        :param fxk: value of the function in x
        """
        xk = x[k]
        m = int(k * (k + 3.0) / 2.0)
        self.C[m] = fxk
        for d in range(1, k + 1):
            xkd = xk - x[k - d]
            for s in range(0, min([d, p]) + 1):
                m = m - 1
                n = m + d
                if s == 0:
                    self.C[m] = self.C[n] + xk * (self.C[n - k - 1] - self.C[n]) / xkd
                elif s == d:
                    self.C[m] = (self.C[n + 1] - self.C[n - k]) / xkd
                else:
                    self.C[m] = self.C[n] + (
                            xk * (self.C[n - k - 1] - self.C[n]) + (self.C[n + 1] - self.C[n - k])) / xkd
            if d > p:
                m = m - d + p

    def __dip__(self, p: int, kmax: int, f: LambdaType, x: list, eps: float) -> tuple:
        """
        Calculates the df(s) = (s-1)-th derivative of f at O , for s = 1 .. p+1. Uses k-th degree polynomial for some k
        satisfying p <= k <= kmax. If the relative change in df(s) from k - 1 to k is less than eps then this determines
         k and success is true. Otherwise k = kmax and success is false.
        :param p:  maximum derivative to calculate; p <= length(x) <= kmax
        :param kmax: maximum degree of the interpolating polynomial
        :param f: function to derive
        :param x: array of values around 0
        :param eps: desired tolerance
        :return: array of derivatives at 0
        """
        self.C = mp.zeros(1, int(kmax * (kmax + 3) / 2) + 1)
        df = mp.zeros(1, p + 1)

        for k in range(0, kmax + 1):
            self.__update__(int(k), int(p), x, f(x[int(k)]))
            if k < p:
                continue
            self.r = mp.mpf(1)
            for s in range(0, p + 1):
                if self.r:
                    self.r = mp.mpf(mp.fabs(self.C[k - s] - df[s]) <= eps * mp.fabs(self.C[k - s]))
                df[s] = self.C[k - s]
            if self.r:
                break

        for s in range(1, p + 1):
            df[s] = mp.factorial(s) * df[s]

        return df

    def derive_at(self, p: int, f: LambdaType, x: float, eps: float, intrv: list) -> tuple:
        """
        Derives f p times in x with desired tolerance eps around the interval intrv. Intrv is an interval around 0, the
        function will take care of the shifting.
        :param p: maximum derivative to calculate;
        :param f: function to derive
        :param x: point in which to derive
        :param eps: desired tolerance
        :param intrv: interval around 0
        :return: array of derivatives at 0
        """
        fun = lambda t: f(t + x)
        npoints = len(intrv)

        df = self.__dip__(p, npoints - 1, fun, intrv, eps)

        if not self.r:
            warnings.warn("Derivation failed to reach precision goal.", RuntimeWarning)

        return df


if __name__ == "__main__":
    # function to test
    f = lambda t: 6 * mp.exp(-6 * t) + t + mp.log(0.2 * t + 4) - 2 ** t - mp.erf(t)

    # Real derivatives, for testing
    f_d = []
    f_d.append(lambda t: 1 / (5 * (t / 5 + 4)) - 36 * mp.exp(-6 * t) - 2 ** t * mp.log(2) - (
            2 * mp.exp(-t ** 2)) / mp.pi ** (1 / 2) + 1)
    f_d.append(lambda t: 216 * mp.exp(-6 * t) - 1 / (25 * (t / 5 + 4) ** 2) - 2 ** t * mp.log(2) ** 2 + (
            4 * t * mp.exp(-t ** 2)) / mp.pi ** (1 / 2))
    f_d.append(lambda t: 2.0 / (125 * (t / 5 + 4) ** 3) - 1296 * mp.exp(-6 * t) + (4 * mp.exp(-t ** 2)) / mp.pi ** (
            1 / 2) - 2 ** t * mp.log(2) ** 3 - (8 * t ** 2 * mp.exp(-t ** 2)) / mp.pi ** (1 / 2))
    f_d.append(lambda t: 7776 * mp.exp(-6 * t) - 6.0 / (625 * (t / 5 + 4) ** 4) - 2 ** t * mp.log(2) ** 4 - (
            24 * t * mp.exp(-t ** 2)) / mp.pi ** (1 / 2) + (16 * t ** 3 * mp.exp(-t ** 2)) / mp.pi ** (1 / 2))

    nPoints = 100  # points around 0
    p = 4  # Calculate all the derivatives until the 4th
    x = -1  # Deriving in -1

    N = NevilleDiff()
    df = N.derive_at(p, f, x, 1e-6, mp.linspace(-0.1, 0.1, nPoints))

    f_dx = [f(0)]
    for funf in f_d:
        f_dx.append(funf(x))

    # Plot
    plt.figure()

    plt.plot(range(0, p + 1), df, 'rx:', label="Neville")
    plt.plot(range(0, p + 1), f_dx, 'k*-.', label="Analytical")
    plt.xlabel("order of derivative")
    plt.legend()

    plt.show()

from sympy import Symbol, symbols, solve, dsolve, Eq, lambdify
from sympy.physics.mechanics import dynamicsymbols, init_vprinting
import matplotlib.pyplot as plt
import scipy as sp

init_vprinting()

# Symbols definition
t = Symbol('t')
C1, C2 = symbols('C1 C2')

u, v, teta, psi = dynamicsymbols('u v theta psi')

symrep1 = (
'm I_d I_p Omega k_x1 k_x2 k_y1 k_y2 a b')
m, Id, Ip, omega, kx1, kx2, ky1, ky2, a, b = symbols(symrep1, positive=True)
symrep2 = (
'k_xT k_xC k_xR k_yT k_yC k_yR k_T k_C k_R')
kxt, kxc, kxr, kyt, kyc, kyr, kt, kc, kr = symbols(symrep2, positive=True)

# Equations of motion (see 3.4 of 'Dynamics of rotating machines')
eq1 = m*(u.diff(t, t)) + (kx1 + kx2)*u + (-a*kx1 + b*kx2)*psi
eq2 = m*(v.diff(t, t)) + (ky1 + ky2)*v + (a*ky1 - b*ky2)*teta
eq3 = Id*(teta.diff(t, t)) + Ip*omega*(psi.diff(t)) + (a*ky1 - b*ky2)*v + (a**2*ky1 + b**2*ky2)*teta
eq4 = Id*(psi.diff(t, t)) - Ip*omega*(teta.diff(t)) + (-a*kx1 + b*kx2)*u + (a**2*kx1 + b**2*kx2)*psi

system = [eq1, eq2, eq3, eq4]


class RigidRotor(object):
    """
    This class creates a rotor object that has as attribute
    the equations of motion of a rigid rotor (see 3.4 of 'Dynamics
    of rotating machines').
    To see the equations, first create a rotor object:
    >>rotor = Rigid_Rotor()
    Then you can access the equations of motion with:
    >> rotor.model
    After a rotor is created, its parameters can be modified to
    obtain a specified model.
    If numerical values are furnished, the equations of motion can
    be solved and values for each time can be obtained. An orbit
    can also be plotted.
    """

    sys_of_eqs = system

    def __init__(self):
        self.model = system

    def define_model(self, parameters):
        """
        This method is used to define the model parameters.
        Each parameter has to be given in a tuple inside a list.
        The tuple is written as follows: (old parameter, new parameter).
        If you want to change the mass for example: (m, 100). This will
        change the symbol m with the value 100.

        Parameters
        ----------
        m: float
            Mass of the rotor
        Id: float
            The diametral moment of inertia
        Ip: float
            The polar moment of inertia
        omega: float
            Rotor speed
        kx1: float
            Stiffness of the first support in the x direction
        kx2: float
            Stiffness of the second support in the x direction
        ky1: float
            Stiffness of the first support in the y direction
        ky2: float
            Stiffness of the second support in the y direction
        a: float
            Distance from the first support to the rotor center of mass
        b: float
            Distance from the second support to the rotor center of mass
        Returns
        ----------
        It will update the rotor model with the furnished parameters.

        Examples:
        Obtain equations for a rigid rotor with isotropic flexible supports:
        >>> rot1 = RigidRotor()
        >>> # subs1_ is used to simplify the equation
        >>> subs1_ = [(kx1 + kx2, kxt),
        ...           (ky1 + ky2, kyt),
        ...           (-a*kx1 + b*kx2, kxc),
        ...           (-a*ky1 + b*ky2, kyc),
        ...           (a**2*kx1 + b**2*kx2, kxr),
        ...           (a**2*ky1 + b**2*ky2, kyr)]
        >>> rot1.define_model(subs1_)
        >>> # Now we use subs_iso1 to define the isotropic supports:
        >>> subs_iso1 = [(kxt, kt),
        ...              (kyt, kt),
        ...              (kxc, kc),
        ...              (kyc, kc),
        ...              (kxr, kr),
        ...              (kyr, kr)]
        >>> rot1.define_model(subs_iso1)
        >>> # Now we can take a look at the equations:
        >>> rot1.model
        [k_C*psi(t) + k_T*u(t) + m*Derivative(u(t), t, t), -k_C*theta(t) + \
k_T*v(t) + m*Derivative(v(t), t, t), I_d*Derivative(theta(t), t, t) \
+ I_p*Omega*Derivative(psi(t), t) - k_C*v(t) + k_R*theta(t), I_d*Derivative(psi(t), t, t) \
- I_p*Omega*Derivative(theta(t), t) + k_C*u(t) + k_R*psi(t)]

        """
        new_sys = []
        for eq in self.model:
            eq_ = eq.subs(parameters)
            new_sys.append(eq_)
            self.model = new_sys

    def solve_model(self, ics, d_ics):
        model_soln = []
        for eq in self.model:
            model_soln.append(dsolve(eq))

        def solve_constants(eq, ics, d_ics):
            udiff = Eq(d_ics[0][1], eq.rhs.diff(t))
            C_1 = solve(eq.subs(ics), {C1, C2})
            C_2 = solve(udiff.subs(t, 0), {C1, C2})
            consts = {}
            consts.update(C_1[0])
            consts.update(C_2[0])
            return eq.subs(consts)

        model_soln_f =[]
        for eq in enumerate(model_soln[:len(ics)]):
            soln = solve_constants(eq[1], ics[eq[0]], d_ics[eq[0]])
            model_soln_f.append(soln)

        self.model_response = model_soln_f
        self.x = lambdify(t, model_soln_f[0].rhs, 'numpy')
        self.y = lambdify(t, model_soln_f[1].rhs, 'numpy')

    def orbit(self):
        t1 = sp.linspace(0, 0.2, 1000)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(self.x(t1[0]), self.y(t1[0]), 'bo')
        ax1.plot(self.x(t1[:240]), self.y(t1[:240]))
        return fig1, plt.show()


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)

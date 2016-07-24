from ROTODINAMICA import *

rotor = RigidRotor()


def test_equations_of_motion():
    assert str(rotor.sys_of_eqs[0]) == 'm*Derivative(u(t), t, t) + (k_x1 + k_x2)*u(t)' \
                                       ' + (-a*k_x1 + b*k_x2)*psi(t)'
    assert str(rotor.sys_of_eqs[1]) == 'm*Derivative(v(t), t, t) + (k_y1 + k_y2)*v(t)' \
                                       ' + (a*k_y1 - b*k_y2)*theta(t)'
    assert str(rotor.sys_of_eqs[2]) == 'I_d*Derivative(theta(t), t, t) + I_p*Omega*Derivative(psi(t), t)' \
                                       ' + (a*k_y1 - b*k_y2)*v(t) + (a**2*k_y1 + b**2*k_y2)*theta(t)'
    assert str(rotor.sys_of_eqs[3]) == 'I_d*Derivative(psi(t), t, t) - I_p*Omega*Derivative(theta(t), t) +' \
                                       ' (-a*k_x1 + b*k_x2)*u(t) + (a**2*k_x1 + b**2*k_x2)*psi(t)'


def test_substitution_of_parameters():
    subs1_ = [(kx1 + kx2, kxt),
              (ky1 + ky2, kyt),
              (-a * kx1 + b * kx2, kxc),
              (-a * ky1 + b * ky2, kyc),
              (a ** 2 * kx1 + b ** 2 * kx2, kxr),
              (a ** 2 * ky1 + b ** 2 * ky2, kyr)]
    rotor.define_model(subs1_)
    assert str(rotor.model[0]) == 'k_xC*psi(t) + k_xT*u(t) + m*Derivative(u(t), t, t)'
    assert str(rotor.model[1]) == '-k_yC*theta(t) + k_yT*v(t) + m*Derivative(v(t), t, t)'
    assert str(rotor.model[2]) == 'I_d*Derivative(theta(t), t, t) + I_p*Omega*Derivative(psi(t), t)' \
                                  ' - k_yC*v(t) + k_yR*theta(t)'
    assert str(rotor.model[3]) == 'I_d*Derivative(psi(t), t, t) - I_p*Omega*Derivative(theta(t), t) ' \
                                  '+ k_xC*u(t) + k_xR*psi(t)'


def test_solve_model():
    subs_iso1 = [(kxt, kt),
                 (kyt, kt),
                 (kxc, kc),
                 (kyc, kc),
                 (kxr, kr),
                 (kyr, kr)]
    subs3511 = [(Ip * omega, 0), (kc, 0)]
    example3511 = [(m, 122.68),
                   (Ip, 0.6134),
                   (Id, 2.8625),
                   (kt, 2 * 10 ** 6),
                   (kr, 125 * 10 ** 3),
                   (kc, 0)]
    ics = [[(u, 0.001), (t, 0)],
           [(v, 0.0005), (t, 0)]]
    d_ics = [[(u, 0.030), (t, 0)],
             [(v, 0), (t, 0)]]
    rotor.define_model(subs_iso1)
    rotor.define_model(subs3511)
    rotor.define_model(example3511)
    rotor.solve_model(ics, d_ics)
    assert rotor.x(1) == -0.00022042617312504603
    assert rotor.y(1) == -0.00021614830593165351


"""This is a function bank. Here are different functions used to iterate stuff
bla bla

def foo(z, c):
    return z + c


"""
import cmath

def power2(z, c):
    return z*z + c


def power3(z, c):
    return z*z*z + c


def power4(z, c):
    return z*z*z*z + c


def power5(z, c):
    return z*z*z*z*z + c


def exp_power3(z, c):
    return cmath(z*z*z) + c


def frac_z_lnz(z, c):
    return ((z*z+z)/cmath.ln(z)) + c

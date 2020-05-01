import numpy as np
import dolfin

from pulse import dolfin_utils


def test_get_constant():

    for value_size in (1, 2):
        for value_rank in (0, 1):

            vals = np.zeros(value_size)
            constant = dolfin_utils.get_constant(
                val=1, value_size=value_size, value_rank=value_rank
            )
            constant.eval(vals, np.zeros(3))
            assert np.all((vals == 1))

            assert isinstance(constant, dolfin.Constant)


def test_map_vector_field():
    pass


def test_base_expression():
    pass


def test_regional_paramter():
    pass


def test_mixed_parameter():
    pass


if __name__ == "__main__":
    test_get_constant()

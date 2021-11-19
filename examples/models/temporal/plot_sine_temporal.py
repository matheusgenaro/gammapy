r"""
.. _sine-temporal-model:

Sinusoidal temporal model
=======================

This model parametrises a time model of sinusoidal modulation.

.. math:: F(t) = 1 + amp * sin(omega*(t-t_{ref}))

"""


# %%
# Example plot
# ------------
# Here is an example plot of the model:

import numpy as np
from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt
# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:
from gammapy.modeling.models import (
    Models,
    PowerLawSpectralModel,
    SineTemporalModel,
    SkyModel,
)

time_range = [Time.now(), Time.now() + 16 * u.d]
omega = np.pi/4. * u.rad/u.day
sine_model = SineTemporalModel(amp=0.5, omega=omega, t_ref=(time_range[0].mjd-0.1)*u.d)
sine_model.plot(time_range)
plt.grid(which="both")


model = SkyModel(
    spectral_model=PowerLawSpectralModel(),
    temporal_model=sine_model,
    name="sine-model",
)
models = Models([model])

print(models.to_yaml())

import numpy as np
from numpy.f2py.auxfuncs import isscalar


class GRBClass:
    def __init__(self, **kwargs):
        self.z = kwargs.get('z')
        self.Epeak = kwargs.get('Epeak')
        self.Eiso_standard = kwargs.get('Eiso')
        self.theta = kwargs.get('theta')

        self.LCDM = kwargs.get('LCDM')
        self.wowaCDM = kwargs.get('wowaCDM')
        self.cosmos = self.LCDM
        self.model = "LambdaCDM"

        # if self.LCDM is not None and self.wowaCDM is None:
        #     self.cosmos = self.LCDM
        #     self.model = "LambdaCDM"
        # elif self.wowaCDM is not None and self.LCDM is None:
        #     self.cosmos = self.wowaCDM
        #     self.model = "w0waCDM"
        # else:
        #     raise KeyError("You didn't provide a universe or you provide multiple models.")
        #todo: implement this to change bewteen models

    def bc(self, energy):
        """
        Returns the barycenter corrected scalar or array values of Epeak, Eiso or Egamma for proper fitting.
        :param energy: energy scalar or array (Epeak, Eiso, Egamma)
        :return: barycenter corrected scalar or array values of Epeak, Eiso or Egamma for proper fitting.
        """
        if isscalar(energy):
            # a point is its own barycenter
            return energy
        barycenter = np.sum(energy) / len(energy)
        return energy - barycenter

    def isotropic_equivalent_energy(self, **kwargs):
        """
        Returns the equivalent isotropic energy for a given cosmological framework.
        :return: Eiso(Omega)
        """
        d_L = self.cosmos.luminosity_distance(self.z)
        Eiso = self.Eiso_standard * (d_L / kwargs.get('DL0'))
        return Eiso

    def collimated_corrected_energy(self):
        """
        Computes the collimated corrected energy for a given isotropic equivalent energy.
        :return: Egamma
        """
        collimation_factor = 1 - np.cos(self.theta)
        return self.isotropic_equivalent_energy() * collimation_factor
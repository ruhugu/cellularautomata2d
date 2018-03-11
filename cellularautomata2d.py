# -*- coding: utf-8 -*-
from __future__ import (print_function, division, 
                        absolute_import, unicode_literals)
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

class CellAutomata2D(object):
    """ General class for celullar automata in a 2D lattice.

    """

    def __init__(self, xlen, ylen, pbc=False, dtype=int, show_cbar=True):
        # Store the lattice parameters
        self.xlen = xlen
        self.ylen = ylen
        self.pbc = pbc
        self.size = self.xlen*self.ylen
        self._show_cbar = show_cbar

        # Lattice length taking into account boundary conditions
        # and number of cells in each side (margins) used to 
        # simulate the boundaries
        if self.pbc: 
            self._xlen_bc = self.xlen
            self._ylen_bc = self.ylen
            self._xmargin = 0
            self._ymargin = 0
        else:
            self._xlen_bc = self.xlen + 2
            self._ylen_bc = self.ylen + 2
            self._xmargin = 1
            self._ymargin = 1
        self._xlimit = self._xlen_bc - self._xmargin
        self._ylimit = self._ylen_bc - self._ymargin
        # Store lattice indexes corresponding to the "true" lattice
        self._latt_idx = (slice(self._xmargin, self._xlimit),
                          slice(self._ymargin, self._ylimit))
    
        # Create the lattice
        self.dtype = dtype
        self._latt_bc = np.zeros((self._xlen_bc, self._ylen_bc), dtype=self.dtype)
        #self.latt = np.zeros((xlen, ylen), dtype=dtype)

        # Create vectors for the implementation of the boundary conditions.
        self._bcx = np.zeros(xlen + 2, dtype=int)
        self._bcy = np.zeros(ylen + 2, dtype=int)
        if self.pbc:
            self._bcx[0] = self.xlen - 1
            self._bcx[self.xlen + 1] = 0
            for i in range(1, self.xlen + 1):
                self._bcx[i] = i - 1

            self._bcy[0] = self.ylen - 1
            self._bcy[self.ylen + 1] = 0
            for i in range(1, self.ylen + 1):
                self._bcy[i] = i - 1
        else:

            self._bcx[0] = 0;
            self._bcx[self.xlen + 1] = self.xlen + 1
            for i in range(1, self.xlen + 1):
                self._bcx[i] = i

            self._bcy[0] = 0
            self._bcy[self.ylen + 1] = self.ylen + 1
            for i in range(1, self.ylen + 1):
                self._bcy[i] = i

    @property
    def latt(self):
        return self._latt_bc[self._latt_idx]
    @latt.setter
    def latt(self, value):
        self._latt_bc[self._latt_idx] = value
        

    def _bc(self, i, j):
        """Return the indices taking into account the PBC.

        """
        return (self._bcx[i+1], self._bcy[j+1])

    def resetlattice(self, fillvalue=0):
        """Fill the lattice with the given value.

        """
        self.latt.fill(fillvalue)
        return

    def mass(self):
        """Return the value of the total mass of the system.

        """
        lattmass = self.latt.sum()
        return lattmass

    def _evolvestep(self):
        """Evolve the system one step.

        Returns
        -------
            is_active : bool
                True if the lattice is acvtive and False otherwise.

        """
        # Placeholder
        is_active = False
        return is_active

    def evolve(self, nsteps=0): 
        """Evolve the system in nsteps timesteps.
            
        Parameters
        ----------
            nsteps : int
                Number of steps the system will be evolved.

        Returns
        -------
            is_active : bool
                True if the lattice is active and False otherwise. If the lattice is active but it does not chage (limit cycle of period 1), the function return True.

        """
        is_active = False

        for i in range(nsteps):
            is_active = self._evolvestep()

        return is_active
    
    def measure(j_t):
        """Measure the current state of the lattice and store it.
        
        This is a placeholder for the actual implentation of the method 
        in the chosen model.
        
        Parameters
        ----------
            j_t : int
                Index of the time of measure in ts_measure (check _measure_run).
        """
        pass
        return
        
    def _measure_run(self, ts_measure):
        """Measure at the given times over one run.
        
        Note: this resets the lattice.
    
        Parameters
        ----------
            ts_measure : int array
                Times when measures are taken in timesteps.
        """
        self.init_latt() 
        nmeasures = ts_measure.size

        # Preparatives for the measuring
        self.init_measures()
        
        last_t = 0
        for j_t, t in enumerate(ts_measure):
            # Calculate the number of steps to take 
            nsteps = t - last_t
            self.evolve(nsteps)

            self.measure(j_t)
            last_t = t

        return

    def relax(self, maxtime=10000, stepsize=1):
        """Evolves the system until it relaxes or the maximum time is reached.

        Parameters
        ----------
            maxtime : int
                Maximum number of steps.
            stepsize : int
                Number of steps taken each time.
        
        Returns
        -------
            relaxtime : int
                Number of steps the system took to relax (if stepsize
                is not 1, this will be an approximation). If the system
                does not relax before the maximum number of steps, -1
                will be returned.
            
        """
        relaxtime = 0
        active = True
        while active and (relaxtime < maxtime):
            active = self.evolve(stepsize)
            relaxtime += stepsize*int(active)

        # If system have not relaxed return -1
        if active:
            relaxtime = -1

        return relaxtime
        

    def findlimitcycle(self, maxtime=50):
        history = np.zeros((maxtime+1, self.xlen, self.ylen))

        foundcycle = False
        relaxtime = -1
        period = -1  # This will be returned if no cycle is found

        j_step = 0
        while (not foundcycle) and j_step <= maxtime:
            history[j_step] = self.latt 
            self.evolve(1)
            j_step += 1

            for i in reversed(range(j_step)):
                if (history[i] == self.latt).all(): 
                    foundcycle = True
                    period = j_step - i
                    relaxtime = j_step
                    break

        return period, relaxtime


    def plot(self, size=3):
        """Plot the system configuration. 

        """

        fig, ax = plt.subplots(figsize=(size,size))
        im = ax.imshow(self.latt, cmap=self.cmap, vmin=self.vmincolor, 
                       vmax=self.vmaxcolor, interpolation=None)
        if self._show_cbar:
            cbar = fig.colorbar(im, ax=ax)

        return fig

    def plotevolution(self, nrows, ncols, steps_per_plot=1, size=2.3, 
                      nplots = None):
        """

        """
        if nplots == None:
            nplots = nrows*ncols

        fig, axs = plt.subplots(nrows, ncols, figsize=(size*ncols, size*nrows))

        for i in range(nrows):
            if nrows > 1:
                for j in range(ncols):
                    if (i*ncols + j) < nplots:
                        im = axs[i,j].imshow(self.latt, cmap=self.cmap, 
                                        vmin=self.vmincolor,
                                        vmax=self.vmaxcolor,
                                        interpolation=None)
                        self.evolve(steps_per_plot)
                    else:
                        axs[i,j].axis("off") 
            else:
                for j in range(ncols):
                    im = axs[j].imshow(self.latt, cmap=self.cmap, 
                                    vmin=self.vmincolor, vmax=self.vmaxcolor,
                                    interpolation=None)
                    self.evolve(steps_per_plot)

        fig.tight_layout()

        if nrows > 1: 
            last_idx = (nrows-1, ncols-1)
        else:
            last_idx = ncols-1

        cbar = fig.colorbar(im, ax=axs)
        return fig


    def animate(self, nframes, steps_per_frame=1, frame_interval=300):
         

        def update(i, steps_per_frame, im, self):
            self.evolve(steps_per_frame)
            im.set_array(self.latt)
            return im

        fig, ax = plt.subplots()
        im = ax.imshow(self.latt, cmap=self.cmap, vmin=self.vmincolor, 
                    vmax=self.vmaxcolor, interpolation=None)

        if self._show_cbar:
            cbar = fig.colorbar(im, ax=ax)

        anim = animation.FuncAnimation(fig, update, frames=nframes, 
                                       blit=False, fargs=(steps_per_frame,
                                       im, self))
        return anim


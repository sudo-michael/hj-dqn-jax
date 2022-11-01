import odp
from odp.solver import HJSolver
from odp.dynamics import DubinsCar
from odp.Grid import Grid
from odp.Shapes import CylinderShape
from odp.Plots import PlotOptions
import  numpy as np

grid = Grid(np.array([-1.0, -1.0, -np.pi]), np.array([1.0, 1.0, np.pi]), 3, np.array([101, 101, 101]), [2])
ivf = CylinderShape(grid, [2], np.zeros(3), 0.5)

dt = 0.05
tau = np.arange(start=0, stop=1.0 + dt, step=dt)
car_brt = DubinsCar(uMode='max', dMode='min', wMax=1.5)
po = PlotOptions(do_plot=True, plot_type="3d_plot", plotDims=[0,1,2], slicesCut=[])

compMethods_brt = { "TargetSetMode": "minVWithV0"}
if __name__ in "__main__":
    result = HJSolver(car_brt, grid, ivf, tau, compMethods_brt, po, saveAllTimeSteps=False)
    np.save("./brt.npy", result)
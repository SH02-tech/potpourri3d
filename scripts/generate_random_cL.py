import os
import numpy as np
import potpourri3d as pp3d
import scipy.io
import scipy.sparse

def generate_random_cl():
	"""
	Generate random connection Laplacian to both real and complex values.
	Generated outputs are saved in the current directory as 'real_cl.npy' 
	and 'complex_cl.npy' (and its corresponding matlab version format).
	"""
	P = np.random.random((1000,3))
	solver = pp3d.PointCloudHeatSolver(P)

	cL = solver.get_connection_laplacian()
	cL_real = solver.get_real_connection_laplacian()

	os.makedirs('scripts/sample', exist_ok=True)

	scipy.sparse.save_npz('scripts/sample/real_cl', cL_real)
	scipy.sparse.save_npz('scripts/sample/complex_cl', cL)
	scipy.io.savemat('scripts/sample/real_cl.mat', {'real_cl': cL_real})
	scipy.io.savemat('scripts/sample/complex_cl.mat', {'complex_cl': cL})

if __name__ == '__main__':
	generate_random_cl()

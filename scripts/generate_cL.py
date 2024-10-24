import os
import argparse
import numpy as np
import potpourri3d as pp3d
import scipy.io
import scipy.sparse

import trimesh

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

	return cL, cL_real

def generate_cl(point_cloud):
	"""
	Generate connection Laplacian from a given point cloud.
	"""
	solver = pp3d.PointCloudHeatSolver(point_cloud)

	cL = solver.get_connection_laplacian()
	cL_real = solver.get_real_connection_laplacian()

	return cL, cL_real

def generate_report(cL, cL_real):
	os.makedirs('scripts/sample', exist_ok=True)

	scipy.sparse.save_npz('scripts/sample/real_cl', cL_real)
	scipy.sparse.save_npz('scripts/sample/complex_cl', cL)
	scipy.io.savemat('scripts/sample/real_cl.mat', {'real_cl': cL_real})
	scipy.io.savemat('scripts/sample/complex_cl.mat', {'complex_cl': cL})

def is_hermitian(A):
    # Calculate the conjugate transpose of A
    A_conj_transpose = np.conjugate(A.T)
    
    # Check if A is Hermitian: A == A^H
    if np.allclose(A, A_conj_transpose):
        return True, 0.0
    else:
		# Calculate the Frobenius norm of the difference A - A^H
        num_elems = len(A.nonzero()[0])
        distance = np.linalg.norm(A - A_conj_transpose, ord='fro') / num_elems
        return False, distance

if __name__ == '__main__':

	arg_parser = argparse.ArgumentParser(description='Generate cL and cL_real')

	arg_parser.add_argument('input', type=str, help='Path to the input point cloud.')

	args = arg_parser.parse_args()

	P = trimesh.load(args.input).vertices
	cL, cL_real = generate_cl(P)
	# cL, cL_real = generate_random_cl()

	is_hermitian, distance = is_hermitian(cL.toarray())

	print(f'Is cL Hermitian? {is_hermitian}')
	print(f'Distance between cL and its conjugate transpose: {distance}')

	generate_report(cL, cL_real)
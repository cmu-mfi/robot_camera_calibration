import numpy as np
import cv2
from calibration.calibrations import *

SAMPLE_SIZE = 10

def get_ideal_sample_size(method):
    '''
    Calculate the ideal sample size for each calibration method
    '''
    
    print(f"Method: {method}")

    errors = []
    std = []
    for sample_size in range(3, 50):
        error_avg = 0
        error_set = []
        for i in range(100):
            T_base2target_set, T_camera2target_set, T_base2camera = get_training_set(sample_size)
            T_base2camera_calculated = solve_rigid_transformation(T_base2target_set, T_camera2target_set, method)
            error = calculate_reprojection_error(T_base2target_set, T_camera2target_set, T_base2camera_calculated)
            error_set.append(error)
        error_avg = np.mean(error_set)
        error_std = np.std(error_set)
        errors.append(error_avg)
        std.append(error_std)
        
    min_error = min(errors)
    min_error_index = errors.index(min_error)
    max_error = max(errors)
    max_error_index = errors.index(max_error)
    print(f"Min. error: {min_error}, std: {std[min_error_index]} sample size: {min_error_index+3}")
    print(f"Max. error: {max_error}, std: {std[max_error_index]} sample size: {max_error_index+3}")    

def test2():
    '''
    Generate 
        (A) a random set of transformation matrices (robot_base to target) or T_base2target_set
        (B) a random transformation matrix (robot_base to camera) or T_base2camera
    and calculate
        (C) a random set of transformation matrices (camera to target) or T_camera2target_set
    
    We use (A) and (C) above to test various approaches to calculate (B), the transformation matrix from robot_base to camera
    
    Assumption:
        1. 3D space
        2. target is on gripper, and T_gripper2target is identity matrix
        
    Note: 
        T_a2b represents frame 'b' in frame 'a', or
        tranforms a point from frame 'b' to frame 'a'
    '''
    print("Hello from test.py")
        
    # generate T_base2target_set
    T_base2target_set = []
    for i in range(SAMPLE_SIZE):
        T = get_random_transformation_matrix()
        T_base2target_set.append(T)
    
    # generate T_base2camera
    T_base2camera = get_random_transformation_matrix()
    print('T_base2camera=\n',T_base2camera)
    
    # calculate T_camera2target_set
    T_camera2target_set = []
    for i in range(SAMPLE_SIZE):
        T = np.dot(np.linalg.inv(T_base2camera), T_base2target_set[i])
        T_camera2target_set.append(T)
    
    # baseline
    avg_error, std_error = calculate_reprojection_error(T_base2target_set, T_camera2target_set, T_base2camera)        
    print(f"\nAvg. reprojection error (baseline): {avg_error}, std. error: {std_error}")    
    
    # method 1: one shot calculation
    T_base2camera_calculated = np.dot(T_base2target_set[0], np.linalg.inv(T_camera2target_set[0]))
    avg_error, std_error = calculate_reprojection_error(T_base2target_set, T_camera2target_set, T_base2camera_calculated)
    print(f"\nAvg. reprojection error (method 1a): {avg_error}, std. error: {std_error}")
    print('T_base2camera_calculated (method 1a)=\n',T_base2camera_calculated)
    T_base2camera_calculated = solve_rigid_transformation(T_base2target_set, T_camera2target_set, method="ONE_SAMPLE_ESTIMATE")
    avg_error, std_error = calculate_reprojection_error(T_base2target_set, T_camera2target_set, T_base2camera_calculated)        
    print(f"\nAvg. reprojection error (method 1b): {avg_error}, std. error: {std_error}")
    print('T_base2camera_calculated (method 1b)=\n',T_base2camera_calculated)    
    
    # method 2: solve_rigid_transformation
    t_camera2target = np.array([T[:3,3] for T in T_camera2target_set])
    t_base2target = np.array([T[:3,3] for T in T_base2target_set])
    T_base2camera_calculated = solve_rigid_transformation_bowen(t_camera2target, t_base2target)
    avg_error, std_error = calculate_reprojection_error(T_base2target_set, T_camera2target_set, T_base2camera_calculated)        
    print(f"\nAvg. reprojection error (method 2): {avg_error}, std. error: {std_error}")
    print('T_base2camera_calculated (method 2)=\n',T_base2camera_calculated)
    T_base2camera_calculated = solve_rigid_transformation(T_base2target_set, T_camera2target_set, method="SVD_ALGEBRAIC")
    avg_error, std_error = calculate_reprojection_error(T_base2target_set, T_camera2target_set, T_base2camera_calculated)               
    print(f"\nAvg. reprojection error (method 2b): {avg_error}, std. error: {std_error}")
    print('T_base2camera_calculated (method 2b)=\n',T_base2camera_calculated) 
    
    # method 3: opencv library. 
    # Details: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
    T_target2base_set = [np.linalg.inv(T) for T in T_base2target_set]
    R_target2base = [T[:3,:3] for T in T_target2base_set]
    t_target2base = [T[:3,3] for T in T_target2base_set]
    
    R_camera2target = [T[:3,:3] for T in T_camera2target_set]
    t_camera2target = [T[:3,3] for T in T_camera2target_set]
    
    # CALIB_HAND_EYE_TSAI 
    R_camera2base, T_base2camera = cv2.calibrateHandEye(R_target2base, t_target2base, 
                                                        R_camera2target, t_camera2target,  
                                                        cv2.CALIB_HAND_EYE_TSAI)
    T_base2camera_calculated = np.eye(4)
    T_base2camera_calculated[:3,:3] = R_camera2base
    T_base2camera_calculated[:3,3] = np.squeeze(T_base2camera)
    avg_error, std_error = calculate_reprojection_error(T_base2target_set, T_camera2target_set, T_base2camera_calculated)
    print(f"\nAvg. reprojection error (method 3 TSAI): {avg_error}, std. error: {std_error}")    
    print('T_base2camera_calculated (method 3 CALIB_HAND_EYE_TSAI )=\n',T_base2camera_calculated)
    T_base2camera_calculated = solve_rigid_transformation(T_base2target_set, T_camera2target_set, method="CALIB_HAND_EYE_TSAI")
    avg_error, std_error = calculate_reprojection_error(T_base2target_set, T_camera2target_set, T_base2camera_calculated)              
    print(f"\nAvg. reprojection error (method 3b TSAI): {avg_error}, std. error: {std_error}")
    print('T_base2camera_calculated (method 3b TSAI)=\n',T_base2camera_calculated) 

def test1():
    '''
    Generate 
        (A) a random set of transformation matrices (robot_base to target) or T_base2target_set
        (B) a random transformation matrix (robot_base to camera) or T_base2camera
    and calculate
        (C) a random set of transformation matrices (camera to target) or T_camera2target_set
    
    We use (A) and (C) above to test various approaches to calculate (B), the transformation matrix from robot_base to camera
    
    Assumption:
        1. 3D space
        2. target is on gripper, and T_gripper2target is identity matrix
        
    Note: 
        T_a2b represents frame 'b' in frame 'a', or
        tranforms a point from frame 'b' to frame 'a'
    '''
    print("Hello from test.py")
    
    # generate T_base2target_set
    T_base2target_set = []
    for i in range(SAMPLE_SIZE):
        T = get_random_transformation_matrix()
        T_base2target_set.append(T)
    
    # generate T_base2camera
    T_base2camera = get_random_transformation_matrix()
    print('T_base2camera=\n',T_base2camera)
    
    # calculate T_camera2target_set
    T_camera2target_set = []
    for i in range(SAMPLE_SIZE):
        T = np.dot(np.linalg.inv(T_base2camera), T_base2target_set[i])
        T_camera2target_set.append(T)
        
    # calculate T_base2camera using T_base2target_set and T_camera2target_set
    T_base2camera_calculated = []
    
    # method 1: one shot calculation
    T_base2camera_calculated = np.dot(T_base2target_set[0], np.linalg.inv(T_camera2target_set[0]))
    avg_error, std_error = calculate_reprojection_error(T_base2target_set, T_camera2target_set, T_base2camera_calculated)
    print(f"\nAvg. reprojection error (method 1): {avg_error}, std. error: {std_error}")
    print('T_base2camera_calculated (method 1)=\n',T_base2camera_calculated)
    
    # method 2: solve_rigid_transformation
    t_camera2target = np.array([T[:3,3] for T in T_camera2target_set])
    t_base2target = np.array([T[:3,3] for T in T_base2target_set])
    T_base2camera_calculated = solve_rigid_transformation_bowen(t_camera2target, t_base2target)
    avg_error, std_error = calculate_reprojection_error(T_base2target_set, T_camera2target_set, T_base2camera_calculated)
    print(f"\nAvg. reprojection error (method 2): {avg_error}, std. error: {std_error}")
    print('T_base2camera_calculated (method 2)=\n',T_base2camera_calculated)
    
    # method 2b: solve_rigid_transformation shobhit
    t_camera2target = np.array([T[:3,3] for T in T_camera2target_set])
    t_base2target = np.array([T[:3,3] for T in T_base2target_set])
    T_base2camera_calculated = solve_rigid_transformation_shobhit(t_camera2target, t_base2target)
    avg_error, std_error = calculate_reprojection_error(T_base2target_set, T_camera2target_set, T_base2camera_calculated)
    print(f"\nAvg. reprojection error (method 2): {avg_error}, std. error: {std_error}")
    print('T_base2camera_calculated (method 2)=\n',T_base2camera_calculated)    
    
    # method 3: opencv library. 
    # Details: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
    T_target2base_set = [np.linalg.inv(T) for T in T_base2target_set]
    R_target2base = [T[:3,:3] for T in T_target2base_set]
    t_target2base = [T[:3,3] for T in T_target2base_set]
    
    R_camera2target = [T[:3,:3] for T in T_camera2target_set]
    t_camera2target = [T[:3,3] for T in T_camera2target_set]
    
    # CALIB_HAND_EYE_TSAI 
    R_camera2base, T_base2camera = cv2.calibrateHandEye(R_target2base, t_target2base, 
                                                        R_camera2target, t_camera2target,  
                                                        cv2.CALIB_HAND_EYE_TSAI)
    T_base2camera_calculated = np.eye(4)
    T_base2camera_calculated[:3,:3] = R_camera2base
    T_base2camera_calculated[:3,3] = np.squeeze(T_base2camera)
    avg_error, std_error = calculate_reprojection_error(T_base2target_set, T_camera2target_set, T_base2camera_calculated)
    print(f"\nAvg. reprojection error (method 3 TSAI): {avg_error}, std. error: {std_error}")    
    print('T_base2camera_calculated (method 3 CALIB_HAND_EYE_TSAI )=\n',T_base2camera_calculated)
    
    # CALIB_HAND_EYE_ANDREFF 
    R_camera2base, T_base2camera = cv2.calibrateHandEye(R_target2base, t_target2base, 
                                                        R_camera2target, t_camera2target,  
                                                        cv2.CALIB_HAND_EYE_ANDREFF)
    T_base2camera_calculated = np.eye(4)
    T_base2camera_calculated[:3,:3] = R_camera2base
    T_base2camera_calculated[:3,3] = np.squeeze(T_base2camera)
    avg_error, std_error = calculate_reprojection_error(T_base2target_set, T_camera2target_set, T_base2camera_calculated)
    print(f"\nAvg. reprojection error (method 3 ANDREFF): {avg_error}, std. error: {std_error}")
    print('T_base2camera_calculated (method 3 CALIB_HAND_EYE_ANDREFF )=\n',T_base2camera_calculated)
    
    # CALIB_HAND_EYE_PARK  
    R_camera2base, T_base2camera = cv2.calibrateHandEye(R_target2base, t_target2base, 
                                                        R_camera2target, t_camera2target,  
                                                        cv2.CALIB_HAND_EYE_PARK )
    T_base2camera_calculated = np.eye(4)
    T_base2camera_calculated[:3,:3] = R_camera2base
    T_base2camera_calculated[:3,3] = np.squeeze(T_base2camera)
    avg_error, std_error = calculate_reprojection_error(T_base2target_set, T_camera2target_set, T_base2camera_calculated)
    print(f"\nAvg. reprojection error (method 3 PARK): {avg_error}, std. error: {std_error}")
    print('T_base2camera_calculated (method 3 CALIB_HAND_EYE_PARK  )=\n',T_base2camera_calculated)

def get_random_transformation_matrix():
    '''
    Generate a random transformation matrix
    '''
    T = np.eye(4)
    
    # rotation matrix
    mat = np.random.rand(3,3)
    q,r = np.linalg.qr(mat)
    if np.linalg.det(q) < 0:
        q[:,0] *= -1
    T[:3,:3] = q
    
    # translation vector
    coords = np.random.randn(3)
    T[:3,3] = (coords / np.linalg.norm(coords)) * np.random.rand(1) * 10
    
    return T  

def get_training_set(sample_size):
        
    # generate T_base2target_set
    T_base2target_set = []
    for i in range(sample_size):
        T = get_random_transformation_matrix()
        T_base2target_set.append(T)
    
    # generate T_base2camera
    T_base2camera = get_random_transformation_matrix()
    
    # calculate T_camera2target_set
    T_camera2target_set = []
    for i in range(sample_size):
        T = np.dot(np.linalg.inv(T_base2camera), T_base2target_set[i])
        T_camera2target_set.append(T)

    return T_base2target_set, T_camera2target_set, T_base2camera
    
def solve_rigid_transformation_bowen(inpts, outpts):
    """
    Takes in two sets of corresponding points, returns the rigid transformation matrix from the first to the second.
    """

    assert inpts.shape == outpts.shape
    inpts, outpts = np.copy(inpts), np.copy(outpts)
    inpt_mean = inpts.mean(axis=0)
    outpt_mean = outpts.mean(axis=0)
    outpts -= outpt_mean
    inpts -= inpt_mean

    X = inpts.T
    Y = outpts.T
    
    print(X.shape)

    covariance = np.dot(X, Y.T)
    U, s, V = np.linalg.svd(covariance)
    S = np.diag(s)
    
    assert np.allclose(covariance, np.dot(U, np.dot(S, V)))
    V = V.T
    idmatrix = np.identity(3)
    idmatrix[2, 2] = np.linalg.det(np.dot(V, U.T))
    R = np.dot(np.dot(V, idmatrix), U.T)
    t = outpt_mean.T - np.dot(R, inpt_mean)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

def solve_rigid_transformation_shobhit(A, B):
    """
    Takes in two sets of corresponding points, returns the rigid transformation matrix from the first to the second.
    Solve for R,t in following equation:
    B = R*A + t
    """

    assert A.shape == B.shape
    A, B = np.copy(A), np.copy(B)
    
    print(A.shape)
    
    # Calculate centroids, and eliminate translation
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    A -= centroid_A
    B -= centroid_B
        
    # Calculate covariance matrix and SVD
    H = np.dot(A.T, B)
    U, s, V = np.linalg.svd(H)
    V = V.T
    
    # Correct for reflection case
    d = np.linalg.det(np.dot(U, V.T))
    I = np.eye(3)
    if d < 0:
        I[2,2] = -1
        
    # Estimate rotation and translation
    R = np.dot(U, np.dot(I, V.T))    
    t = centroid_B - np.dot(R, centroid_A)

    # Return transformation matrix
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
            
    return T


def calculate_reprojection_error_bowen(T_a2t_set, T_b2t_set, T_a2b):
    errors = []
    assert len(T_a2t_set) == len(T_b2t_set)
    
    # Calculate error using transformation projections
    for i in range(len(T_b2t_set)):
        T_a2t_calc = np.dot(T_a2b, T_b2t_set[i])
        error = np.linalg.norm(T_a2t_set[i] - T_a2t_calc)
        errors.append(error)
    
    # Calculate error using pose projections
    # T_a2b = np.linalg.inv(T_b2a)
    # for i in range(len(T_t2a_set)):
    #     pose_target_in_frame_b = T_t2b_set[i][:,3]
    #     pose_target_in_frame_a = np.dot(T_a2b, pose_target_in_frame_b)
    #     error = np.linalg.norm(T_t2a_set[i][:,3] - pose_target_in_frame_a)
    #     errors.append(error)
        
    # Compute average error
    avg_error = np.mean(errors)
    std_error = np.std(errors)
    return avg_error, std_error

if __name__ == "__main__":
    # print("=======test1=======\n")
    test1()
    #print("=======test2=======\n")
    #test2()
    # print("=======get_ideal_sample_size=======\n")
    # get_ideal_sample_size("SVD_ALGEBRAIC")
    # get_ideal_sample_size("CALIB_HAND_EYE_TSAI")
    # get_ideal_sample_size("CALIB_HAND_EYE_ANDREFF")

import numpy as np
import pyvista as pv
import copy
import cosserat

# returns the matrix log of a SO3 rotation (i.e. maps SO3 --> so3)
def MatLog_SO3(mat):
    cos_theta = 0.5*mat.trace() - 0.5
    # theta = np.acos(np.max(-1.0, np.min(1.0, cos_theta)))
    theta = np.acos(cos_theta)

    K = np.array([mat[2,1]-mat[1,2], mat[0,2]-mat[2,0], mat[1,0]-mat[0,1]])

    # if theta is close to 0
    if np.abs(theta) < 1e-8:
        return 0.5 * K
    elif np.abs(theta - np.pi) < 1e-8:
        # 180-degree rotation case
        # extract rotation axis from R+I
        max_idx = np.argmax( [mat[0,0]+1, mat[1,1]+1, mat[2,2]+1] )
        axis = np.array( [mat[max_idx,0], mat[max_idx,1], mat[max_idx,2]] )
        axis[max_idx] += 1
        axis /= np.linalg.norm(axis)
        return theta * axis
    else:
        # general case
        return (0.5 * theta / np.sin(theta)) * K


    vee = Vee_SO3(mat - mat.transpose())
    
    if (np.abs(mat.trace() - 3) < 1e-8):
        omega = 0.5 * (1 + 1/6 * theta**2 + 7/360 * theta**4) * vee
    else:
        omega = theta / (2*np.sin(theta)) * vee
    
    return omega

# wlt::Matrix<T, 3, 1> Log(const wlt::Matrix<T, 3, 3>& R) {
#     const T tr = R.trace();
#     const T cos_theta = (tr - 1) * 0.5;
    
#     // Clamp cos_theta to [-1,1] to handle numerical errors
#     T theta = std::acos(std::max(T(-1), std::min(T(1), cos_theta)));
    
#     // Extract skew-symmetric part
#     wlt::Matrix<T, 3, 1> K(R(2, 1) - R(1, 2),
#                           R(0, 2) - R(2, 0),
#                           R(1, 0) - R(0, 1));
    
#     if (std::abs(theta) < T(1e-10)) {
#         // Near identity rotation
#         return T(0.5) * K;
#     } else if (std::abs(theta - M_PI) < T(1e-10)) {
#         // 180-degree rotation case
#         // Need to extract rotation axis from R+I
#         wlt::Matrix<T, 3, 1> axis;
#         T max_diag = R(0,0) + 1;
#         int max_idx = 0;
#         for (int i = 1; i < 3; ++i) {
#             if (R(i,i) + 1 > max_diag) {
#                 max_diag = R(i,i) + 1;
#                 max_idx = i;
#             }
#         }
#         axis = wlt::Matrix<T, 3, 1>(R(max_idx,0), R(max_idx,1), R(max_idx,2));
#         axis += wlt::Matrix<T, 3, 1>::Unit(max_idx);
#         axis.normalize();
#         return theta * axis;
#     } else {
#         // General case
#         return (T(0.5) * theta / std::sin(theta)) * K;
#     }
# }

# returns the 3x3 skew-symmetric matrix for a 3-vector
def Skew3(v):
    return np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])


# maps skew matrix into 3-vector
def Vee_SO3(mat):
    return np.array([mat[2,1], mat[0,2], mat[1,0]])

# computes e^([w]theta), the matrix exponential of a so3 exponential coordinates vector, yielding a rotation matrix
# omega is a 3x1 exponential coordinates vector
# theta is a scalar, magnitude of the rotation
def MatExp_so3(omega, theta):
    skew = Skew3(omega)
    return np.eye(3) + np.sin(theta)*skew + (1 - np.cos(theta))*np.matmul(skew,skew)

# computes the matrix exponential of a twist (se3), yielding a 4x4 homogeneous transformation matrix
# V is a 6x1 twist vector, with rotational components coming first
def MatExp_se3(V):
    omega = V[0:3]
    v = V[3:6]
    theta = np.linalg.norm(omega)
    # special case: if theta = 0, rotation part of transformation is identity
    if theta < 1e-12:
        R = np.eye(3)
        p = v
    else:
        # normalize the twist according to theta
        omega /= theta
        v /= theta
        # compute rotational part of transformation matrix
        R = MatExp_so3(omega, theta)
        # compute translational part of transformation matrix
        skew = Skew3(omega)
        G = np.eye(3) * theta + (1 - np.cos(theta))*skew + (theta - np.sin(theta))*np.matmul(skew,skew)
        p = np.matmul(G,v)
    
    # put the transformation matrix together from rotational and translational parts
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    return T

def centroid(points):
    # fit a plane onto the cross-section points with least squares
    # and project points onto this plane (the deformed mesh cross section may no longer be exactly planar)
    _,planar_points = fitPlane(points)

    face = list(range(len(points)))     # create the face
    face.insert(0, len(points))     

    poly_data = pv.PolyData(planar_points, face)   # create the PolyData object

    # triangulate using poly data
    poly_data.triangulate(inplace=True)

    # calculate the centroid using the formula: sum(Ci * Ai) / sum(Ai) for each triangle in the cross-section
    numerator = 0
    total_area = 0
    for i in range(0,len(poly_data.faces), 4):
        face = poly_data.faces[i+1:i+4]
        tri = poly_data.points[face]
        A = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
        cen = (tri[0] + tri[1] + tri[2]) / 3
        numerator += cen * A
        total_area += A

    centroid = numerator/total_area
    return centroid

def area(points):
    # fit a plane onto the cross-section points with least squares
    # and project points onto this plane (the deformed mesh cross section may no longer be exactly planar)
    _,planar_points = fitPlane(points)

    face = list(range(len(points)))     # create the face
    face.insert(0, len(points))     

    poly_data = pv.PolyData(planar_points, face)   # create the PolyData object

    # triangulate using poly data
    poly_data.triangulate(inplace=True)

    # calculate the centroid using the formula: sum(Ci * Ai) / sum(Ai) for each triangle in the cross-section
    total_area = 0
    for i in range(0,len(poly_data.faces), 4):
        face = poly_data.faces[i+1:i+4]
        tri = poly_data.points[face]
        A = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
        total_area += A

    return total_area

# fits a plane to the vertices using least-squares
# assumes vertices is a (nx3) array
def fitPlane(vertices):
    np_vertices = np.matrix(vertices)
    num_vertices, _ = np_vertices.shape

    # https://stackoverflow.com/a/44315488
    # formulate least squares problem as "tall" Ax = b where x is the unknown plane coeffs
    # ax + by + c = z, we are solving for [a,b,c]
    # rows of [xi,yi,1] make up A and rows of [zi] make up b 
    A = np.ones((num_vertices, 3))
    A[:,:2] = vertices[:,:2]
    b = vertices[:,2]

    # solve overconstrained Ax=b using the left pseudo-inverse
    fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.transpose(), A)), A.transpose()), b)

    # get the plane normal - for the plane ax + by + c = z it will be [a, b, -1]
    plane_normal = np.array([fit[0], fit[1], -1])
    plane_normal /= np.linalg.norm(plane_normal)    # normalize the plane normal
    # choose arbitrary plane origin
    plane_origin = np.array([0, 0, fit[2]])

    # calculate the corrections needed for each vertex to be moved along the plane normal to the plane
    vertex_corrections = np.zeros(vertices.shape)
    for i in range(vertices.shape[0]):
        # get perpinduclar distance from point to plane: 
        dist = np.dot((vertices[i,:] - plane_origin), plane_normal)
        vertex_corrections[i,:] = -plane_normal * dist
    
    # apply the corrections
    planar_vertices = vertices + vertex_corrections

    # return the plane coefficients and the planar vertices
    return fit, planar_vertices

# takes in the data loaded from a COMSOL output txt file (loaded with numpy.loadtxt) and the undeformed mesh
# and finds correspondance between the vertices from the COMSOL output and the .stl file to appropriately deform each vertex of the input .stl file
def getDeformedMeshFromComsolData(comsol_data_path, undeformed_mesh):
    deformed_mesh = copy.copy(undeformed_mesh)
    comsol_data = np.loadtxt(comsol_data_path, comments='%')
    # assuming no duplicate vertices
    for row in comsol_data:
        orig_v = row[:3]
        new_v = row[3:]
        
        sub = undeformed_mesh.vertices - orig_v
        index = np.argwhere(np.linalg.norm(sub, axis=1) < 1e-5)
        deformed_mesh.vertices[index.item()] = new_v
    
    return deformed_mesh

# takes in an undeformed .stl FEA mesh and deforms it according to the Nastran output files
#   undeformed_mesh is generated using FNO reader and converting the .NAS file to a .stl file
#   undeformed_nodes_filename is the filename of the (cleaned) undeformed nodes .csv file, output from FNO reader converting the .NAS file to .txt file (outputting only the coordinates)
#   node_displacements_filename is the filename of the (cleaned) displacements .csv file, output from FNO reader converting the .FNO file to table .txt file 
def getDeformedMeshFromNastranData(undeformed_mesh, undeformed_nodes_filename, node_displacements_filename):
    deformed_mesh = copy.copy(undeformed_mesh)

    # load the original nodes from the .csv file
    nodes_loaded_data = np.genfromtxt(undeformed_nodes_filename, delimiter=',', dtype=None) # use dtype=None to read different types
    undeformed_nodes = np.zeros((len(nodes_loaded_data), 3), dtype=float)
    # the nodal coordinates are in columns 3, 4, and 5
    for i,row in enumerate(nodes_loaded_data):
        vert = np.array([row[3], row[4], row[5]])
        undeformed_nodes[i,:] = vert
    
    # load nodal displacements from the .csv file
    disp_loaded_data = np.genfromtxt(node_displacements_filename, delimiter=',', dtype=None)
    node_displacements = np.zeros((len(disp_loaded_data), 3), dtype=float)
    # the nodal displacement coordinates are in columns 4, 5, and 6
    for i,row in enumerate(disp_loaded_data):
        disp = np.array([row[4], row[5], row[6]])
        node_displacements[i,:] = disp
    
    # match up .stl vertices with the nodes from the .csv file (the .csv file contains ALL nodes, including internal nodes)
    for i,vert in enumerate(undeformed_mesh.vertices):
        sub = undeformed_nodes - vert
        # find the index of the match
        index = np.argwhere(np.linalg.norm(sub, axis=1) < 1e-5)
        # and apply the appropriate displacement
        deformed_mesh.vertices[i] += node_displacements[index.item(),:]

    return deformed_mesh

def getNodesNastran(undeformed_nodes_filename, node_displacements_filename):
    # load the original nodes from the .csv file
    nodes_loaded_data = np.genfromtxt(undeformed_nodes_filename, delimiter=',', dtype=None) # use dtype=None to read different types
    undeformed_nodes = np.zeros((len(nodes_loaded_data), 3), dtype=float)
    # the nodal coordinates are in columns 3, 4, and 5
    for i,row in enumerate(nodes_loaded_data):
        vert = np.array([row[3], row[4], row[5]])
        undeformed_nodes[i,:] = vert

    # load nodal displacements from the .csv file
    disp_loaded_data = np.genfromtxt(node_displacements_filename, delimiter=',', dtype=None)
    node_displacements = np.zeros((len(nodes_loaded_data), 3), dtype=float)
    # the nodal displacement coordinates are in columns 4, 5, and 6
    for i,row in enumerate(disp_loaded_data):
        disp = float(row[6])
        node_num = int(row[5]) - 1
        if "T1" in row[3]:
            node_displacements[node_num, 0] = disp
        elif "T2" in row[3]:
            node_displacements[node_num, 1] = disp
        elif "T3" in row[3]:
            node_displacements[node_num, 2] = disp
    
    deformed_nodes = undeformed_nodes + node_displacements
    return undeformed_nodes, deformed_nodes

def loadRodFromFile(filename):
    with open(filename, 'r') as file:
        data = file.read()
        data_arr = data.split("\n")

    N = int(data_arr[0])
    length = float(data_arr[1])
    E = float(data_arr[2])
    nu = float(data_arr[3])
    cs_type = data_arr[4]
    cs_rx = float(data_arr[5])
    cs_ry = float(data_arr[6])
    state = np.array(data_arr[7:]).astype(float)

    # if the state does not have a,b,c, add default values
    if len(state) < 3*N + 6*(N-1):
        a_0 = np.ones(N)      # a = 1 when cross-section is undeformed
        b_0 = np.ones(N)      # b = 1 when cross-section is undeformed
        c_0 = np.zeros(N)     # c = 0 when cross-section is undeformed
        state = np.hstack( (a_0, b_0, c_0, state) )

    if cs_type == "Ellipse":
        cross_section = cosserat.AnalyticalEllipseCrossSection(cs_rx, cs_ry)
    elif cs_type == "Rect":
        cross_section = cosserat.AnalyticalRectCrossSection(cs_rx*2, cs_ry*2)
    
    if len(state) > 3*N + 6*(N-1):
        rod = cosserat.LinearDeformationCosseratRod(N, length, cross_section, E, nu)
    else:
        rod = cosserat.CosseratRod(N, length, cross_section, E, nu)

    deformed_rod = copy.deepcopy(rod)
    deformed_rod.Z = state

    return (rod, deformed_rod)

    
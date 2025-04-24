import numpy as np
import pyvista as pv
import copy

# returns the matrix log of a SO3 rotation (i.e. maps SO3 --> so3)
def MatLog_SO3(mat):
    theta = np.arccos(0.5*mat.trace() - 0.5)

    # if theta is 0, return 0
    if np.abs(theta) < 1e-10:
        omega = np.array([0,0,0])
        return omega


    vee = Vee_SO3(mat - mat.transpose())
    
    if (np.abs(mat.trace() - 3) < 1e-8):
        omega = 0.5 * (1 + 1/6 * theta**2 + 7/360 * theta**4) * vee
    else:
        omega = theta / (2*np.sin(theta)) * vee
    
    return omega

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


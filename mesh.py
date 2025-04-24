import numpy as np
import pyvista as pv
import trimesh as tm

import utils

# a class representing a point that is an interpolation of two points
# the points are either vertex indices or InterpolatedPoints themselves (i.e. you can nest InterpolatedPoints)
# by storing vertex indices rather than the vertices themselves, we can get the same "material" point on different meshes
#  i.e. we can track a point between the undeformed and deformed meshes (assuming both meshes have the same vertices and vertex numbering)
class InterpolatedPoint:
    # v1 - index of vertex 1
    # v2 - index of vertex 2
    # s - interpolation parameter in [0,1]
    def __init__(self, v1, v2, s):

        # for consistency, we "sort" v1 and v2 such that v1 is always lower than v2 in terms of the vertex index
        if (type(v1) == InterpolatedPoint):
            v1_ind = v1.getLowestVertexIndex()
        else:
            v1_ind = v1
        
        if (type(v2) == InterpolatedPoint):
            v2_ind = v2.getLowestVertexIndex()
        else:
            v2_ind = v2

        # sort v1 and v2
        if v1_ind < v2_ind:
            self.v1 = v1
            self.v2 = v2
            self.s = s
        else:
            self.v1 = v2
            self.v2 = v1
            self.s = 1-s

    # helper method for sorting children vertices - returns the lowest vertex index in all of the nested children
    def getLowestVertexIndex(self):
        v = self.v1
        while (type(v) == InterpolatedPoint):
            v = v.v1

        return v

    # get the actual point in the mesh from the matrix of vertices using interpolation 
    def getPoint(self, vertices):
        if type(self.v1) == InterpolatedPoint:
            p1 = self.v1.getPoint(vertices)
        else:
            p1 = vertices[self.v1, :]
        
        if type(self.v2) == InterpolatedPoint:
            p2 = self.v2.getPoint(vertices)
        else:
            p2 = vertices[self.v2, :]

        return p1 + (p2 - p1) * self.s
    
    # check for equality between two InterpolatedPoints
    def __eq__(self, other):
        # there's probably a better way to do this but oh well
        if (self.v1 == other.v1 and self.v2 == other.v2 and abs(self.s - other.s) < 1e-12):
            return True
        
        if (self.v1 == other.v1 and abs(self.s) < 1e-12 and abs(other.s) < 1e-12):
            return True
        
        if (self.v2 == other.v2 and abs(1 - self.s) < 1e-12 and abs(1 - other.s) < 1e-12):
            return True
        
        if (self.v1 == other.v2 and abs(self.s) < 1e-12 and abs(1 - other.s) < 1e-12):
            return True

        if (self.v2 == other.v1 and abs(1-self.s) < 1e-12 and abs(other.s) < 1e-12):
            return True
        
        return False
            
    
    def __str__(self):
        return f"v1: {self.v1}, v2: {self.v2}, s: {self.s}"
    
class MeshCrossSection:

    def __init__(self, points, x_axis, y_axis):
        self.points = points
        self.x_axis = x_axis
        self.y_axis = y_axis
    
    # given a vertices matrix of the mesh, return the vertices of this cross-section as a PyVista PolyData object
    # cross-section points will be projected so that they are guaranteed to be planar
    def asPolyData(self, vertices):
        points = np.array([p.getPoint(vertices) for p in self.points] )   # get all the 3D points

        # fit a plane onto the cross-section points with least squares
        # and project points onto this plane (the deformed mesh cross section may no longer be exactly planar)
        _,planar_points = utils.fitPlane(points)

        face = list(range(len(points)))     # create the face
        face.insert(0, len(points))     

        poly_data = pv.PolyData(planar_points, face)   # create the PolyData object
        return poly_data
    
    def asPolyData2D(self, vertices):
        points = np.array([p.getPoint(vertices) for p in self.points] )   # get all the 3D points

        # fit a plane onto the cross-section points with least squares
        # and project points onto this plane (the deformed mesh cross section may no longer be exactly planar)
        _,planar_points = utils.fitPlane(points)

        # project planar points to 2D (dot points with x and y axes)
        centroid = self.centroid(vertices)
        x_point = self.x_axis.getPoint(vertices)
        y_point = self.y_axis.getPoint(vertices)
        x_axis = x_point - centroid
        y_axis = y_point - centroid
        unit_x = x_axis / np.linalg.norm(x_axis)
        unit_y = y_axis / np.linalg.norm(y_axis)

        points_2d = []
        for p in planar_points:
            x_coord = np.dot(unit_x, p)
            y_coord = np.dot(unit_y, p)
            points_2d.append([x_coord, y_coord, 0])
        
        points_2d = np.array(points_2d)
        face = list(range(len(points_2d)))
        face.insert(0, len(points))
        poly_data = pv.PolyData(points_2d, face)
        return poly_data

        


    def centroid(self, vertices):
        poly_data = self.asPolyData(vertices)
        return utils.centroid(poly_data.points)
    
    def axes(self, vertices):
        centroid = self.centroid(vertices)

        return [centroid, self.x_axis.getPoint(vertices), self.y_axis.getPoint(vertices)]
        




def _edgePlaneIntersection(edge_p1, edge_p2, plane_normal, plane_point):
    edge_vec = edge_p2 - edge_p1 # vec going from p1 to p2

    dot1 = np.dot(edge_vec, plane_normal)
    dot2 = np.dot((plane_point - edge_p1), plane_normal)


    if abs(dot1) < 1e-12:    # check if edge and plane are parallel
        if abs(dot2) < 1e-12:    # line is contained in the plane
            return [0, 1]
        else:               # line is not contained in the plane - no intersection
            return None
    else:
        d = dot2/dot1   # "distance" for intersection along edge between p1 and p2
        if d >= 0 and d <= 1:   # if d is between 0 (p1) and 1 (p2), we have an intersection on the line segment
            return [d]
        else:   # otherwise no intersection on the line segment
            return None
        
def _facePlaneIntersection(vertices, face, plane_normal, plane_point):
    p1 = vertices[face[0],:]
    p2 = vertices[face[1],:]
    p3 = vertices[face[2],:]

    face_intersections = []
    d12 = _edgePlaneIntersection(p1, p2, plane_normal, plane_point)
    d13 = _edgePlaneIntersection(p1, p3, plane_normal, plane_point)
    d23 = _edgePlaneIntersection(p2, p3, plane_normal, plane_point)

    if d12 is not None:
        for d in d12:
            ip = InterpolatedPoint(face[0], face[1], d)
            if ip not in face_intersections:
                face_intersections.append(ip)
    if d13 is not None:
        for d in d13:
            ip = InterpolatedPoint(face[0], face[2], d)
            if ip not in face_intersections:
                face_intersections.append(ip)
    if d23 is not None:
        for d in d23:
            ip = InterpolatedPoint(face[1], face[2], d)
            if ip not in face_intersections:
                face_intersections.append(ip)

    return face_intersections


## finds the cross-section of the undeformed mesh with a plane parallel to the XY plane at height z
# and then finds the associated cross-section in the deformed mesh
#  (assumes that the undeformed and deformed meshes have the same vertex numbering and faces)
#
# returns the cross-sections as pv.PolyData objects for visualization
def getCrossSectionsMesh(undeformed_mesh, deformed_mesh, z_undeformed):
    

    # find vertices with min and max Z in undeformed mesh - these are our beginning and ends of the rod
    undeformed_min_z = np.min(undeformed_mesh.vertices[:,2])
    undeformed_max_z = np.max(undeformed_mesh.vertices[:,2])
    base_vertices = (undeformed_mesh.vertices[:,2] == undeformed_min_z).nonzero()
    tip_vertices = (undeformed_mesh.vertices[:,2] == undeformed_max_z).nonzero()

    deformed_min_z = np.min(undeformed_mesh.vertices[:,2])
    deformed_base_vertices = (deformed_mesh.vertices[:,2] == deformed_min_z).nonzero()

    # defines a plane parallel to XY plane
    plane_normal = np.array([0,0,1])
    plane_point = np.array([0,0,z_undeformed])

    cross_section_points = []
    cross_section_edges = []
    for f in undeformed_mesh.faces:
        
        face_intersections = _facePlaneIntersection(undeformed_mesh.vertices, f, plane_normal, plane_point)
        if len(face_intersections) == 2:
            cross_section_edges.append(face_intersections)

    # for edge in cross_section_edges:
    #     print(f"p1: {edge[0]};\tp2: {edge[1]}")

    cross_section_points = cross_section_edges.pop(0)
    for i in range(len(cross_section_edges)-1):
        # find next connecting edge
        next_ind = -1
        next_point = -1
        for ind, edge in enumerate(cross_section_edges):
            if edge[0] == cross_section_points[-1]:
                next_ind = ind
                next_point = 1
                break
            elif edge[1] == cross_section_points[-1]:
                next_ind = ind
                next_point = 0
                break
        
        if next_ind == -1:
            print("SOMETHING WENT WRONG! AHHHHH COULDN'T FIND NEXT EDGE TO LINK")
            break
        
        # print(f"next ind: {next_ind}, next_point: {next_point}")
        next_edge = cross_section_edges.pop(next_ind)
        # print(f"next p1: {next_edge[0]};\tnext p2: {next_edge[1]}")
        cross_section_points.append(next_edge[next_point])

    # find intersection between undeformed cross-section and x and y-axes
    points = np.array([p.getPoint(undeformed_mesh.vertices) for p in cross_section_points])
    centroid = utils.centroid(points)
    x_ip = None
    y_ip = None
    for i in range(len(points)-1):
        p_cur = points[i]
        p_next = points[i+1]
        if (p_cur[0] - centroid[0]) * (p_next[0] - centroid[0]) < 0 and (p_cur[1] - centroid[1]) > 0: # one x is positive and one is negative, and y is positive
            # find where intersection with y axis is
            s = -(p_cur[0] - centroid[0]) / (p_next[0] - p_cur[0])
            y_ip = InterpolatedPoint(cross_section_points[i], cross_section_points[i+1], s)
        
        if (p_cur[1] - centroid[1]) * (p_next[1] - centroid[1]) < 0 and (p_cur[0] - centroid[0]) > 0: # one y is positive and one is negative, and x is positive
            # find where intersection with x axis is
            s = -(p_cur[1] - centroid[1]) / (p_next[1] - p_cur[1])
            x_ip = InterpolatedPoint(cross_section_points[i], cross_section_points[i+1], s)

    if x_ip is None or y_ip is None:
        print("SOMETHING WENT WRONG! AHHHHH COULDN't FIND X AND Y AXES OF UNDEFORMED CROSS SECTION!")

    cross_section = MeshCrossSection(cross_section_points, x_ip, y_ip)

    undeformed_poly_data = cross_section.asPolyData(undeformed_mesh.vertices)
    deformed_poly_data = cross_section.asPolyData(deformed_mesh.vertices)

    return cross_section, undeformed_poly_data, deformed_poly_data




def visualize():
    undeformed_mesh = tm.load_mesh("UndeformedMesh.stl")
    deformed_mesh = tm.load_mesh("DeformedMesh.stl")

    deformed_mesh.apply_translation([2000,0,0])

    undeformed_vertices = undeformed_mesh.vertices
    deformed_vertices = deformed_mesh.vertices

    cross_section, undeformed_cross_section, deformed_cross_section = getCrossSectionsMesh(undeformed_mesh, deformed_mesh, 10000)
    undeformed_centroid = cross_section.centroid(undeformed_vertices)
    print(f"undeformed centroid: {undeformed_centroid}")

    deformed_centroid = cross_section.centroid(deformed_vertices)
    print(f"deformed centroid: {deformed_centroid}")

    [undef_origin, undef_x, undef_y] = cross_section.axes(undeformed_vertices)
    [def_origin, def_x, def_y] = cross_section.axes(deformed_vertices)

    print(undef_origin)
    print(undef_x)
    print(undef_y)

    plotter = pv.Plotter()
    plotter.add_mesh(undeformed_mesh, color=[255,0,0], opacity=0.5)
    plotter.add_mesh(deformed_mesh, color=[0,0,255], opacity=0.5)
    plotter.add_mesh(undeformed_cross_section, color='red', opacity=0.7, show_edges=True, edge_color='black', line_width=3)
    plotter.add_mesh(deformed_cross_section, color='blue', opacity=0.7, show_edges=True, edge_color='black', line_width=3)
    plotter.add_floor(pad=1)

    plotter.add_lines(np.array([undef_origin, undef_x]))
    plotter.add_lines(np.array([undef_origin, undef_y]))
    plotter.add_lines(np.array([def_origin, def_x]))
    plotter.add_lines(np.array([def_origin, def_y]))
    plotter.show()

    # small_sphere = pv.Sphere().compute_normals()
    # inflated_points = (
    #     small_sphere.points + 0.1 * small_sphere.point_data['Normals']
    # )
    # larger_sphere = pv.PolyData(
    #     inflated_points, faces=small_sphere.GetPolys()
    # )
    # plotter = pv.Plotter()
    # _ = plotter.add_mesh(small_sphere, color='red', show_edges=True)
    # _ = plotter.add_mesh(
    #     larger_sphere, color='blue', opacity=0.3, show_edges=True
    # )
    # plotter.show()
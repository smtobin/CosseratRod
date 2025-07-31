import numpy as np
import scipy.optimize as opt
import math
import pyvista as pv

import mesh
import utils

class AnalyticalEllipseCrossSection:
    def __init__(self, rx, ry):
        self.rx = rx
        self.ry = ry
        self.Ix = np.pi / 4 * rx * ry**3
        self.Iy = np.pi / 4 * ry * rx**3
        self.A0 = np.pi * rx * ry

    def meshPoints(self, resolution=30):
        points = np.zeros((3,resolution))
        for i in range(resolution):
            angle = i * 2*math.pi / (resolution)
            points[:,i]= [self.rx*math.cos(angle), self.ry*math.sin(angle), 0]
        
        return points

    # given the 3x3 C of the deformation, get the deformed area of the cross-section
    def deformedArea(self, C):
        return np.linalg.det(C) * self.A0

class AnalyticalRectCrossSection:
    def __init__(self, sx, sy):
        self.rx = sx/2
        self.ry = sy/2
        self.Ix = sx * sy**3 / 12
        self.Iy = sy * sx**3 / 12
        self.Ix2y2 = sx**3 / 12 * sy**3 / 12    # area integral of x^2 * y^2
        self.Ix2 = sx * sy**5 / 80              # area integral of y^4
        self.Iy2 = sy * sx**5 / 80              # area integral of x^4
        self.A0 = sx*sy
    
    def meshPoints(self):
        points = np.zeros((3, 4))
        points[:,0] = [-self.rx, -self.ry, 0]
        points[:,1] = [self.rx, -self.ry, 0]
        points[:,2] = [self.rx, self.ry, 0]
        points[:,3] = [-self.rx, self.ry, 0]

        return points

    # given the 3x3 C of the deformation, get the deformed area of the cross-section
    def deformedArea(self, C):
        return np.linalg.det(C) * self.A0
    
    
class AnalyticalCavityCrossSection:
    def __init__(self, xs_outer, xs_inner, xa, ya):
        self.xs_outer = xs_outer
        self.xs_inner = xs_inner
        self.xa = xa
        self.ya = ya

        self.cx = (0*xs_outer.A0 - xa*xs_inner.A0) / (xs_outer.A0 - xs_inner.A0)
        self.cy = (0*xs_outer.A0 - ya*xs_inner.A0) / (xs_outer.A0 - xs_inner.A0)

        # moment of inertia ABOUT CENTROID
        self.Ix = (xs_outer.Ix + xs_outer.A0 * (0 - self.cy)**2) - (xs_inner.Ix + xs_inner.A0 * (ya - self.cy)**2)
        self.Iy = (xs_outer.Iy + xs_outer.A0 * (0 - self.cx)**2) - (xs_inner.Iy + xs_inner.A0 * (xa - self.cx)**2)

        self.A0 = xs_outer.A0 - xs_inner.A0

    # given the 3x3 C of the deformation, get the deformed area of the cross-section
    def deformedArea(self, C):
        return np.linalg.det(C) * self.A0
    
    # given the 3x3 C of the deformation, get the deformed area of the inner cross-section
    def deformedAreaInner(self, C):
        return np.linalg.det(C) * self.xs_inner.A0


# simple class for packaging up a force applied at the tip
#   force = force 3-vector
#   location = the (x,y) location of the force in the undeformed cross-section
#   dead_load = whether or not the load is "dead" - if true, the force is in global coords, otherwise it changes with the orientation of the tip
class AppliedTipForce:
    def __init__(self, force, location, dead_load):
        self.force = force
        self.location = location
        self.dead_load = dead_load

class AppliedTipMoment:
    def __init__(self, moment):
        self.moment = moment

class Tendon:
    def __init__(self, tension, location):
        self.tension = tension
        self.location = location
        self.diameter = 0.03

    def asMesh(self, rod, resolution=30):
        circle_points = np.zeros((3,resolution))
        for i in range(resolution):
            angle = i * 2*math.pi / (resolution)
            circle_points[:,i]= [self.diameter/2*math.cos(angle), self.diameter/2*math.sin(angle), 0]

        unit_dir = np.array([self.location[:2]]) / np.linalg.norm(self.location[:2])
        extra_translation = unit_dir.squeeze() * self.diameter/3
        # extra_translation = [0,0,0]

        node_transforms = rod.nodeTransforms()
        xsections = []
        h = rod.L / (rod.n - 1)
        for i,transform in enumerate(node_transforms):
            
            
            if self.location[2] <= (i-1)*h:
                    break
            elif self.location[2] > (i-1)*h and self.location[2] < i*h:
                T = rod.nodeTransform(self.location[2])
                C = (rod.nodeDistortionMatrix(i-1) + rod.nodeDistortionMatrix(i))/2
            else:
                T = transform
                C = rod.nodeDistortionMatrix(i)

            R = T[0:3, 0:3]
            t = T[0:3, 3]
            transformed_circle_points = np.matmul(R, circle_points) + t[:, np.newaxis]

            x_dir = T[0:3, 0]
            y_dir = T[0:3, 1]

            xy_location = np.array([self.location[0]+extra_translation[0], self.location[1]+extra_translation[1], 0])
            in_plane_translation = np.matmul(R, np.matmul(C, xy_location))
            # transformed_circle_points += in_plane_translation
            # in_plane_translation = x_dir * (self.location[0] + extra_translation[0]) + y_dir * (self.location[1] + extra_translation[1])
            transformed_circle_points += in_plane_translation[:, np.newaxis]
            xsections.append(transformed_circle_points) 

            
        
        # calculate number of 'rings' the tendon has
        n = len(xsections)
        # n = len(node_transforms)
        # create matrix for all vertices in the mesh
        vertices = np.zeros((resolution*n, 3))

        for i,xsection in enumerate(xsections):
            # add the points to the overall matrix
            vertices[resolution*(i):resolution*(i+1),:] = xsection.transpose()

        # create 'side' faces for mesh between each cross section
        faces = np.zeros((resolution*(n-1), 5), dtype=int)
        for i in range(n-1):
            for k in range(resolution):
                p1 = i*resolution + k
                p2 = i*resolution + (k+1)
                p3 = (i+1)*resolution + (k+1)
                p4 = (i+1)*resolution + k
                if (k == resolution - 1):
                    p2 = i*resolution
                    p3 = (i+1)*resolution
                
                faces[i*resolution + k, :] = [4, p1, p2, p3, p4]
        
        # create end faces
        # s=0 end face
        bottom_face = list(range(resolution))
        bottom_face.insert(0, len(bottom_face))

        # s=L end face
        top_face = list(range(resolution*(n-1), resolution*n))
        top_face.insert(0, len(top_face))
        
        # concatenate faces into one array - PyVista reads them in as a flat array
        all_faces = np.hstack([faces.flatten(), bottom_face, top_face])
        
        # create and plot the mesh with PyVista
        surf = pv.PolyData(vertices, all_faces)
        surf.triangulate(inplace=True)

        return surf


###############################################################################
# Cosserat rod with constant cross-sectional deformation
###############################################################################
class CosseratRod:
    def __init__(self, n, L, cross_section, E, nu):
        self.n = n
        self.L = L
        self.cross_section = cross_section

        # set material properties
        self.E = E
        self.nu = nu

        M = E * (1-nu) / ( (1+nu) * (1-2*nu))
        lam = E * nu / ( (1+nu) * (1-2*nu))
        self.M = M
        self.G = E / (2 * (1+nu))       # shear modulus
        self.K = np.matrix([[ M, lam, lam, 0, 0, 0 ],   # stiffness matrix for Cosserat rod
                            [ lam, M, lam, 0, 0, 0 ],
                            [ lam, lam, M, 0, 0, 0 ],
                            [ 0, 0, 0, self.G, 0, 0 ],
                            [ 0, 0, 0, 0, self.G, 0 ],
                            [ 0, 0, 0, 0, 0, self.G ]])
        
        # initialize Z
        # define initial variables for optimization problem (variables in the undeformed state)
        #_ cross-sectional paramaters a, b, c are defined at each of the n nodes along the length
        a_0 = np.ones(n)      # a = 1 when cross-section is undeformed
        b_0 = np.ones(n)      # b = 1 when cross-section is undeformed
        c_0 = np.zeros(n)     # c = 0 when cross-section is undeformed

        # link parameters are defined at the midpoint between the nodes, i.e. there are n-1 links
        v1_0 = np.zeros(n-1)  # v1 = 0 when no loads on rod
        v2_0 = np.zeros(n-1)  # v2 = 0 when no loads on rod
        v3_0 = np.ones(n-1)   # v3 = 1 when no loads on rod
        u1_0 = np.zeros(n-1)  # u1 = 0 when no loads on rod
        u2_0 = np.zeros(n-1)  # u2 = 0 when no loads on rod
        u3_0 = np.zeros(n-1)  # u3 = 0 when no loads on rod

        self.Z = np.hstack([a_0, b_0, c_0, v1_0, v2_0, v3_0, u1_0, u2_0, u3_0])

    def solveOptimizationProblem(self, applied_tip_forces):
        # put some bounds/constraints on Z
        # bounds of (None,None) indicate that a variable has no bounds in the optimization
        Z_bounds = [(None,None)]*self.Z.size # array of None's the size of Z

        # constrain cross-section at base of rod (s=0) to be undeformed
        Z_bounds[0] = (1,1)     # a at s=0 is 1 (undeformed)
        Z_bounds[self.n] = (1,1)     # b at s=0 is 1 (undeformed)
        Z_bounds[2*self.n] = (0,0)   # c at s=0 is 0 (undeformed)

        # constrain cross-section at end of rod (s=L) to be undeformed
        Z_bounds[self.n-1] = (1,1)   # a at s=L is 1 (undeformed)
        Z_bounds[2*self.n-1] = (1,1) # b at s=L is 1 (undeformed)
        Z_bounds[3*self.n-1] = (0,0) # c at s=L is 0 (undeformed)

        # constrain the rest of a, b, and c to be reasonable values (shouldn't be needed)
        # for i in range(1,self.n-1):
        #     Z_bounds[i] = (0, 10)
        #     Z_bounds[self.n + i] = (0, 10)
        #     Z_bounds[2*self.n + i] = (-10, 10)

        # constrain v1, v2, v3 to be reasonable values (shouldn't be needed)
        # for i in range(0, self.n-1):
        #     Z_bounds[3*self.n + i] = (0,0)
        #     Z_bounds[3*self.n + self.n-1 + i] = (0,0)
        #     Z_bounds[3*self.n + 2*(self.n-1) + i] = (0.5,2)

        # constrain u1, u2, u3 to be reasonable values (shouldn't be needed)
        # for i in range(0, self.n-1):
        #     Z_bounds[3*self.n + 3*(self.n-1) + i] = (0,0)
        #     Z_bounds[3*self.n + 4*(self.n-1) + i] = (0,0)
        #     Z_bounds[3*self.n + 5*(self.n-1) + i] = (0,0)

        res = opt.minimize(self._totalEnergy, self.Z, args=(applied_tip_forces), method='slsqp', bounds=Z_bounds, options={'maxiter': 10000, 'disp':True})
        self.Z = res.x

        a =  self.Z[0:self.n]
        b =  self.Z[self.n:2*self.n]
        c =  self.Z[2*self.n:3*self.n]
        v1 = self.Z[3*self.n:4*self.n-1]
        v2 = self.Z[4*self.n-1:5*self.n-2]
        v3 = self.Z[5*self.n-2:6*self.n-3]
        u1 = self.Z[6*self.n-3:7*self.n-4]
        u2 = self.Z[7*self.n-4:8*self.n-5]
        u3 = self.Z[8*self.n-5:9*self.n-6]

        print(f"a: {a}")
        print(f"b: {b}")
        print(f"c: {c}")
    
    def solveOptimizationProblemWithMoments(self, applied_tip_forces, applied_tip_moments):
        # put some bounds/constraints on Z
        # bounds of (None,None) indicate that a variable has no bounds in the optimization
        Z_bounds = [(None,None)]*self.Z.size # array of None's the size of Z

        # constrain cross-section at base of rod (s=0) to be undeformed
        Z_bounds[0] = (1,1)     # a at s=0 is 1 (undeformed)
        Z_bounds[self.n] = (1,1)     # b at s=0 is 1 (undeformed)
        Z_bounds[2*self.n] = (0,0)   # c at s=0 is 0 (undeformed)

        # constrain cross-section at end of rod (s=L) to be undeformed
        Z_bounds[self.n-1] = (1,1)   # a at s=L is 1 (undeformed)
        Z_bounds[2*self.n-1] = (1,1) # b at s=L is 1 (undeformed)
        Z_bounds[3*self.n-1] = (0,0) # c at s=L is 0 (undeformed)

        res = opt.minimize(self._totalEnergyWithMoments, self.Z, args=(applied_tip_forces, applied_tip_moments), method='slsqp', bounds=Z_bounds, options={'maxiter': 10000, 'disp':True})
        self.Z = res.x

        a =  self.Z[0:self.n]
        b =  self.Z[self.n:2*self.n]
        c =  self.Z[2*self.n:3*self.n]
        v1 = self.Z[3*self.n:4*self.n-1]
        v2 = self.Z[4*self.n-1:5*self.n-2]
        v3 = self.Z[5*self.n-2:6*self.n-3]
        u1 = self.Z[6*self.n-3:7*self.n-4]
        u2 = self.Z[7*self.n-4:8*self.n-5]
        u3 = self.Z[8*self.n-5:9*self.n-6]

        print(f"a: {a}")
        print(f"b: {b}")
        print(f"c: {c}")

    def solveOptimizationProblemWithMomentsAndTorsionalCorrection(self, applied_tip_forces, applied_tip_moments, torsional_correction):
        # put some bounds/constraints on Z
        # bounds of (None,None) indicate that a variable has no bounds in the optimization
        Z_bounds = [(None,None)]*self.Z.size # array of None's the size of Z

        # constrain cross-section at base of rod (s=0) to be undeformed
        Z_bounds[0] = (1,1)     # a at s=0 is 1 (undeformed)
        Z_bounds[self.n] = (1,1)     # b at s=0 is 1 (undeformed)
        Z_bounds[2*self.n] = (0,0)   # c at s=0 is 0 (undeformed)

        # constrain cross-section at end of rod (s=L) to be undeformed
        Z_bounds[self.n-1] = (1,1)   # a at s=L is 1 (undeformed)
        Z_bounds[2*self.n-1] = (1,1) # b at s=L is 1 (undeformed)
        Z_bounds[3*self.n-1] = (0,0) # c at s=L is 0 (undeformed)

        res = opt.minimize(self._totalEnergyWithMomentsAndTorsionalCorrection, self.Z, args=(applied_tip_forces, applied_tip_moments, torsional_correction), method='slsqp', bounds=Z_bounds, options={'maxiter': 10000, 'disp':True})
        self.Z = res.x

        a =  self.Z[0:self.n]
        b =  self.Z[self.n:2*self.n]
        c =  self.Z[2*self.n:3*self.n]
        v1 = self.Z[3*self.n:4*self.n-1]
        v2 = self.Z[4*self.n-1:5*self.n-2]
        v3 = self.Z[5*self.n-2:6*self.n-3]
        u1 = self.Z[6*self.n-3:7*self.n-4]
        u2 = self.Z[7*self.n-4:8*self.n-5]
        u3 = self.Z[8*self.n-5:9*self.n-6]

        print(f"a: {a}")
        print(f"b: {b}")
        print(f"c: {c}")

    def solveOptimizationProblemCosserat(self, applied_tip_forces):
        # put some bounds/constraints on Z
        # bounds of (None,None) indicate that a variable has no bounds in the optimization
        Z_bounds = [(None,None)]*self.Z.size # array of None's the size of Z

        # constrain cross-section at base of rod (s=0) to be undeformed
        Z_bounds[0] = (1,1)     # a at s=0 is 1 (undeformed)
        Z_bounds[self.n] = (1,1)     # b at s=0 is 1 (undeformed)
        Z_bounds[2*self.n] = (0,0)   # c at s=0 is 0 (undeformed)

        # constrain cross-section at end of rod (s=L) to be undeformed
        # Z_bounds[self.n-1] = (1,1)   # a at s=L is 1 (undeformed)
        # Z_bounds[2*self.n-1] = (1,1) # b at s=L is 1 (undeformed)
        # Z_bounds[3*self.n-1] = (0,0) # c at s=L is 0 (undeformed)

        res = opt.minimize(self._totalEnergyCosserat, self.Z, args=(applied_tip_forces), method='slsqp', bounds=Z_bounds, options={'maxiter': 10000, 'disp':True})
        self.Z = res.x

        a =  self.Z[0:self.n]
        b =  self.Z[self.n:2*self.n]
        c =  self.Z[2*self.n:3*self.n]
        v1 = self.Z[3*self.n:4*self.n-1]
        v2 = self.Z[4*self.n-1:5*self.n-2]
        v3 = self.Z[5*self.n-2:6*self.n-3]
        u1 = self.Z[6*self.n-3:7*self.n-4]
        u2 = self.Z[7*self.n-4:8*self.n-5]
        u3 = self.Z[8*self.n-5:9*self.n-6]

        print(f"a: {a}")
        print(f"b: {b}")
        print(f"c: {c}")

    def solveOptimizationProblemLinearized(self, applied_tip_forces):
        # put some bounds/constraints on Z
        # bounds of (None,None) indicate that a variable has no bounds in the optimization
        Z_bounds = [(None,None)]*self.Z.size # array of None's the size of Z

        # constrain cross-section at base of rod (s=0) to be undeformed
        Z_bounds[0] = (1,1)     # a at s=0 is 1 (undeformed)
        Z_bounds[self.n] = (1,1)     # b at s=0 is 1 (undeformed)
        Z_bounds[2*self.n] = (0,0)   # c at s=0 is 0 (undeformed)

        # constrain cross-section at end of rod (s=L) to be undeformed
        # Z_bounds[self.n-1] = (1,1)   # a at s=L is 1 (undeformed)
        # Z_bounds[2*self.n-1] = (1,1) # b at s=L is 1 (undeformed)
        # Z_bounds[3*self.n-1] = (0,0) # c at s=L is 0 (undeformed)

        # constrain the rest of a, b, and c to be reasonable values (shouldn't be needed)
        # for i in range(1,self.n-1):
        #     Z_bounds[i] = (0, 10)
        #     Z_bounds[self.n + i] = (0, 10)
        #     Z_bounds[2*self.n + i] = (-10, 10)

        # constrain v1, v2, v3 to be reasonable values (shouldn't be needed)
        # for i in range(0, self.n-1):
        #     Z_bounds[3*self.n + i] = (0,0)
        #     Z_bounds[3*self.n + self.n-1 + i] = (0,0)
        #     Z_bounds[3*self.n + 2*(self.n-1) + i] = (0.5,2)

        # constrain u1, u2, u3 to be reasonable values (shouldn't be needed)
        # for i in range(0, self.n-1):
        #     Z_bounds[3*self.n + 3*(self.n-1) + i] = (0,0)
        #     Z_bounds[3*self.n + 4*(self.n-1) + i] = (0,0)
        #     Z_bounds[3*self.n + 5*(self.n-1) + i] = (0,0)

        res = opt.minimize(self._totalEnergyLinearized, self.Z, args=(applied_tip_forces), method='slsqp', bounds=Z_bounds, options={'maxiter': 10000, 'disp':True})
        self.Z = res.x

        a =  self.Z[0:self.n]
        b =  self.Z[self.n:2*self.n]
        c =  self.Z[2*self.n:3*self.n]
        v1 = self.Z[3*self.n:4*self.n-1]
        v2 = self.Z[4*self.n-1:5*self.n-2]
        v3 = self.Z[5*self.n-2:6*self.n-3]
        u1 = self.Z[6*self.n-3:7*self.n-4]
        u2 = self.Z[7*self.n-4:8*self.n-5]
        u3 = self.Z[8*self.n-5:9*self.n-6]

        print(f"a: {a}")
        print(f"b: {b}")
        print(f"c: {c}")

    def solveOptimizationProblemLinearizedNoBendingCorrection(self, applied_tip_forces):
        # put some bounds/constraints on Z
        # bounds of (None,None) indicate that a variable has no bounds in the optimization
        Z_bounds = [(None,None)]*self.Z.size # array of None's the size of Z

        # constrain cross-section at base of rod (s=0) to be undeformed
        Z_bounds[0] = (1,1)     # a at s=0 is 1 (undeformed)
        Z_bounds[self.n] = (1,1)     # b at s=0 is 1 (undeformed)
        Z_bounds[2*self.n] = (0,0)   # c at s=0 is 0 (undeformed)

        # constrain cross-section at end of rod (s=L) to be undeformed
        # Z_bounds[self.n-1] = (1,1)   # a at s=L is 1 (undeformed)
        # Z_bounds[2*self.n-1] = (1,1) # b at s=L is 1 (undeformed)
        # Z_bounds[3*self.n-1] = (0,0) # c at s=L is 0 (undeformed)

        # constrain the rest of a, b, and c to be reasonable values (shouldn't be needed)
        # for i in range(1,self.n-1):
        #     Z_bounds[i] = (0, 10)
        #     Z_bounds[self.n + i] = (0, 10)
        #     Z_bounds[2*self.n + i] = (-10, 10)

        # constrain v1, v2, v3 to be reasonable values (shouldn't be needed)
        # for i in range(0, n-1):
        #     Z_bounds[3*n + i] = (0,0)
        #     Z_bounds[3*n + n-1 + i] = (0,0)
        #     Z_bounds[3*n + 2*(n-1) + i] = (0.5,2)

        # constrain u1, u2, u3 to be reasonable values (shouldn't be needed)
        # for i in range(0, n-1):
        #     Z_bounds[3*n + 3*(n-1) + i] = (-1,1)
        #     Z_bounds[3*n + 4*(n-1) + i] = (-1,1)
        #     Z_bounds[3*n + 5*(n-1) + i] = (-1,1)

        res = opt.minimize(self._totalEnergyLinearizedNoBendingCorrection, self.Z, args=(applied_tip_forces), method='slsqp', bounds=Z_bounds, options={'maxiter': 10000, 'disp':True})
        self.Z = res.x

        a =  self.Z[0:self.n]
        b =  self.Z[self.n:2*self.n]
        c =  self.Z[2*self.n:3*self.n]
        v1 = self.Z[3*self.n:4*self.n-1]
        v2 = self.Z[4*self.n-1:5*self.n-2]
        v3 = self.Z[5*self.n-2:6*self.n-3]
        u1 = self.Z[6*self.n-3:7*self.n-4]
        u2 = self.Z[7*self.n-4:8*self.n-5]
        u3 = self.Z[8*self.n-5:9*self.n-6]

        print(f"a: {a}")
        print(f"b: {b}")
        print(f"c: {c}")
    
    def solveOptimizationProblemTorsionalCorrection(self, applied_tip_forces):
        # put some bounds/constraints on Z
        # bounds of (None,None) indicate that a variable has no bounds in the optimization
        Z_bounds = [(None,None)]*self.Z.size # array of None's the size of Z

        # constrain cross-section at base of rod (s=0) to be undeformed
        Z_bounds[0] = (1,1)     # a at s=0 is 1 (undeformed)
        Z_bounds[self.n] = (1,1)     # b at s=0 is 1 (undeformed)
        Z_bounds[2*self.n] = (0,0)   # c at s=0 is 0 (undeformed)

        # constrain cross-section at end of rod (s=L) to be undeformed
        # Z_bounds[self.n-1] = (1,1)   # a at s=L is 1 (undeformed)
        # Z_bounds[2*self.n-1] = (1,1) # b at s=L is 1 (undeformed)
        # Z_bounds[3*self.n-1] = (0,0) # c at s=L is 0 (undeformed)

        res = opt.minimize(self._totalEnergyTorsionalCorrection, self.Z, args=(applied_tip_forces), method='slsqp', bounds=Z_bounds, options={'maxiter': 10000, 'disp':True})
        self.Z = res.x

        a =  self.Z[0:self.n]
        b =  self.Z[self.n:2*self.n]
        c =  self.Z[2*self.n:3*self.n]
        v1 = self.Z[3*self.n:4*self.n-1]
        v2 = self.Z[4*self.n-1:5*self.n-2]
        v3 = self.Z[5*self.n-2:6*self.n-3]
        u1 = self.Z[6*self.n-3:7*self.n-4]
        u2 = self.Z[7*self.n-4:8*self.n-5]
        u3 = self.Z[8*self.n-5:9*self.n-6]

        print(f"a: {a}")
        print(f"b: {b}")
        print(f"c: {c}")

    def _totalEnergyCosserat(self, Z, applied_tip_forces):
        h = self.L / (self.n - 1)

        energy = 0
        g = np.eye(4) # current transform along the rod g(s)

        # extract variables
        v1 = Z[3*self.n:4*self.n-1]
        v2 = Z[4*self.n-1:5*self.n-2]
        v3 = Z[5*self.n-2:6*self.n-3]
        u1 = Z[6*self.n-3:7*self.n-4]
        u2 = Z[7*self.n-4:8*self.n-5]
        u3 = Z[8*self.n-5:9*self.n-6]

        # iterate through the links (i.e. midpoints between the nodes)
        J = self.cross_section.Ix + self.cross_section.Iy #0.141 * (self.cross_section.rx*2)**4 # hard-coded for a square
        for i in range(0,self.n-1):

            # compute energy for this segment
            energy += 0.5 * h * (self.G*self.cross_section.A0*v1[i]**2 + self.G*self.cross_section.A0*v2[i]**2 + self.E*self.cross_section.A0*(v3[i] - 1)**2 +
                                 self.E*self.cross_section.Ix*u1[i]**2 + self.E*self.cross_section.Iy*u2[i]**2 + self.G*J*u3[i]**2
                                 )
            # compute energy stored in this segment
            # energy += 0.5 * h * (e0_energy + ex_energy + ey_energy)
            
            # compute the transform from the base to node i
            g = np.matmul(g, utils.MatExp_se3([h*u1[i], h*u2[i], h*u3[i], h*v1[i], h*v2[i], h*v3[i]]))
            
        x_dir = g[0:3, 0]  # tip transform x-direction (a coord is in this direction)
        y_dir = g[0:3, 1]  # tip transform y-direction (b coord is in this direction)
        tip_pos = g[0:3, 3]
        
        for tip_force in applied_tip_forces:
            p = tip_pos + x_dir * tip_force.location[0] + y_dir * tip_force.location[1] # position at end of rod where force is applied

            if tip_force.dead_load:
                force = tip_force.force
            else:
                R = g[0:3, 0:3]
                force = np.matmul(R.transpose(), tip_force.force)   # rotate force from body coords into global frame
                # print(force)

            # print(np.dot(force,p))
            # subtract the applied force from the total internal energy
            energy -= np.dot(force, p)
            
        return energy.item() # use .item() to turn the 1x1 matrix into a scalar
    
    def _totalEnergyLinearizedNoBendingCorrection(self, Z, applied_tip_forces):
        h = self.L / (self.n - 1)

        energy = 0
        g = np.eye(4) # current transform along the rod g(s)

        # extract variables
        a = Z[0:self.n]
        b = Z[self.n:2*self.n]
        c = Z[2*self.n:3*self.n]
        v1 = Z[3*self.n:4*self.n-1]
        v2 = Z[4*self.n-1:5*self.n-2]
        v3 = Z[5*self.n-2:6*self.n-3]
        u1 = Z[6*self.n-3:7*self.n-4]
        u2 = Z[7*self.n-4:8*self.n-5]
        u3 = Z[8*self.n-5:9*self.n-6]

        # iterate through the links (i.e. midpoints between the nodes)
        for i in range(0,self.n-1):
            # get cross-sectional parameters at midpoint between nodes
            a_mid = (a[i] + a[i+1])/2
            b_mid = (b[i] + b[i+1])/2
            c_mid = (c[i] + c[i+1])/2

            # get approximate derivates of a, b, and c
            a_prime = (a[i+1] - a[i])/h
            b_prime = (b[i+1] - b[i])/h
            c_prime = (c[i+1] - c[i])/h

            # compute energy for this segment
            strain_vec = np.array([a_mid - 1, b_mid -1, v3[i] - 1])
            M_mat = self.K[0:3, 0:3]
            energy += 0.5 * h * (self.G*self.cross_section.A0*v1[i]**2 + self.G*self.cross_section.A0*v2[i]**2 + 4*self.G*self.cross_section.A0*c_mid**2 +
                                 self.M*self.cross_section.Ix*u1[i]**2 + self.M*self.cross_section.Iy*u2[i]**2 + self.G*(self.cross_section.Ix + self.cross_section.Iy)*u3[i]**2 +
                                 self.cross_section.A0 * np.matmul(np.matmul(strain_vec, M_mat), strain_vec) +
                                 self.G*self.cross_section.Iy*a_prime**2 + self.G*self.cross_section.Ix*b_prime**2 + self.G*(self.cross_section.Ix + self.cross_section.Iy)*c_prime**2 +
                                 2*self.G*u3[i]*c_prime*(self.cross_section.Iy - self.cross_section.Ix)
                                 )
            # compute energy stored in this segment
            # energy += 0.5 * h * (e0_energy + ex_energy + ey_energy)
            
            # compute the transform from the base to node i
            g = np.matmul(g, utils.MatExp_se3([h*u1[i], h*u2[i], h*u3[i], h*v1[i], h*v2[i], h*v3[i]]))
            
        x_dir = g[0:3, 0]  # tip transform x-direction (a coord is in this direction)
        y_dir = g[0:3, 1]  # tip transform y-direction (b coord is in this direction)
        tip_pos = g[0:3, 3]
        
        for tip_force in applied_tip_forces:
            p = tip_pos + x_dir * tip_force.location[0] + y_dir * tip_force.location[1] # position at end of rod where force is applied

            if tip_force.dead_load:
                force = tip_force.force
            else:
                R = g[0:3, 0:3]
                force = np.matmul(R.transpose(), tip_force.force)   # rotate force from body coords into global frame
                # print(force)

            # print(np.dot(force,p))
            # subtract the applied force from the total internal energy
            energy -= np.dot(force, p)
            
        return energy.item() # use .item() to turn the 1x1 matrix into a scalar

    def _totalEnergyLinearized(self, Z, applied_tip_forces):
        h = self.L / (self.n - 1)

        energy = 0
        g = np.eye(4) # current transform along the rod g(s)

        # extract variables
        a = Z[0:self.n]
        b = Z[self.n:2*self.n]
        c = Z[2*self.n:3*self.n]
        v1 = Z[3*self.n:4*self.n-1]
        v2 = Z[4*self.n-1:5*self.n-2]
        v3 = Z[5*self.n-2:6*self.n-3]
        u1 = Z[6*self.n-3:7*self.n-4]
        u2 = Z[7*self.n-4:8*self.n-5]
        u3 = Z[8*self.n-5:9*self.n-6]

        # iterate through the links (i.e. midpoints between the nodes)
        for i in range(0,self.n-1):
            # get cross-sectional parameters at midpoint between nodes
            a_mid = (a[i] + a[i+1])/2
            b_mid = (b[i] + b[i+1])/2
            c_mid = (c[i] + c[i+1])/2

            # get approximate derivates of a, b, and c
            a_prime = (a[i+1] - a[i])/h
            b_prime = (b[i+1] - b[i])/h
            c_prime = (c[i+1] - c[i])/h

            # compute energy for this segment
            strain_vec = np.array([a_mid - 1, b_mid -1, v3[i] - 1])
            M_mat = self.K[0:3, 0:3]
            energy += 0.5 * h * (self.G*self.cross_section.A0*v1[i]**2 + self.G*self.cross_section.A0*v2[i]**2 + 4*self.G*self.cross_section.A0*c_mid**2 +
                                 self.E*self.cross_section.Ix*u1[i]**2 + self.E*self.cross_section.Iy*u2[i]**2 + self.G*(self.cross_section.Ix + self.cross_section.Iy)*u3[i]**2 +
                                 self.cross_section.A0 * np.matmul(np.matmul(strain_vec, M_mat), strain_vec) +
                                 self.G*self.cross_section.Iy*a_prime**2 + self.G*self.cross_section.Ix*b_prime**2 + self.G*(self.cross_section.Ix + self.cross_section.Iy)*c_prime**2 +
                                 2*self.G*u3[i]*c_prime*(self.cross_section.Iy - self.cross_section.Ix)
                                 )
            # compute energy stored in this segment
            # energy += 0.5 * h * (e0_energy + ex_energy + ey_energy)
            
            # compute the transform from the base to node i
            g = np.matmul(g, utils.MatExp_se3([h*u1[i], h*u2[i], h*u3[i], h*v1[i], h*v2[i], h*v3[i]]))
            
        x_dir = g[0:3, 0]  # tip transform x-direction (a coord is in this direction)
        y_dir = g[0:3, 1]  # tip transform y-direction (b coord is in this direction)
        tip_pos = g[0:3, 3]
        
        for tip_force in applied_tip_forces:
            p = tip_pos + x_dir * tip_force.location[0] + y_dir * tip_force.location[1] # position at end of rod where force is applied

            if tip_force.dead_load:
                force = tip_force.force
            else:
                R = g[0:3, 0:3]
                force = np.matmul(R.transpose(), tip_force.force)   # rotate force from body coords into global frame
                # print(force)

            # print(np.dot(force,p))
            # subtract the applied force from the total internal energy
            energy -= np.dot(force, p)
            
        return energy.item() # use .item() to turn the 1x1 matrix into a scalar

    def _totalEnergy(self, Z, applied_tip_forces):
        h = self.L / (self.n - 1)

        energy = 0
        g = np.eye(4) # current transform along the rod g(s)

        # extract variables
        a = Z[0:self.n]
        b = Z[self.n:2*self.n]
        c = Z[2*self.n:3*self.n]
        v1 = Z[3*self.n:4*self.n-1]
        v2 = Z[4*self.n-1:5*self.n-2]
        v3 = Z[5*self.n-2:6*self.n-3]
        u1 = Z[6*self.n-3:7*self.n-4]
        u2 = Z[7*self.n-4:8*self.n-5]
        u3 = Z[8*self.n-5:9*self.n-6]

        # iterate through the links (i.e. midpoints between the nodes)
        for i in range(0,self.n-1):
            # get cross-sectional parameters at midpoint between nodes
            a_mid = (a[i] + a[i+1])/2
            b_mid = (b[i] + b[i+1])/2
            c_mid = (c[i] + c[i+1])/2

            # get approximate derivates of a, b, and c
            a_prime = (a[i+1] - a[i])/h
            b_prime = (b[i+1] - b[i])/h
            c_prime = (c[i+1] - c[i])/h

            # compute energy for this segment
            k11 = self.E*(self.cross_section.Ix*b_mid**2 + self.cross_section.Iy*c_mid**2)
            k22 = self.E*(self.cross_section.Iy*a_mid**2 + self.cross_section.Ix*c_mid**2)
            k12 = -self.E*(self.cross_section.Iy*a_mid*c_mid + self.cross_section.Ix*b_mid*c_mid)
            k33 = self.G*self.cross_section.Iy*(a_mid**2 + c_mid**2) + self.G*self.cross_section.Ix*(b_mid**2 + c_mid**2)
            K_mat = np.matrix([ [k11, k12, 0], [k12, k22, 0], [0, 0, k33] ])
            u_vec = np.array([u1[i], u2[i], u3[i]])
            strain_vec = np.array([a_mid - 1, b_mid -1, v3[i] - 1])
            M_mat = self.K[0:3, 0:3]
            energy += 0.5 * h * (self.G*self.cross_section.A0*v1[i]**2 + self.G*self.cross_section.A0*v2[i]**2 + 4*self.G*self.cross_section.A0*c_mid**2 +
                                 np.matmul(np.matmul(u_vec, K_mat), u_vec) +
                                 self.cross_section.A0 * np.matmul(np.matmul(strain_vec, M_mat), strain_vec) +
                                 self.G*self.cross_section.Iy*a_prime**2 + self.G*self.cross_section.Ix*b_prime**2 + self.G*(self.cross_section.Ix + self.cross_section.Iy)*c_prime**2 +
                                 2*self.G*u3[i]*( (a_mid*c_prime - a_prime*c_mid)*self.cross_section.Iy - (b_mid*c_prime - b_prime*c_mid)*self.cross_section.Ix)
                                 )
            # compute energy stored in this segment
            # energy += 0.5 * h * (e0_energy + ex_energy + ey_energy)
            
            # compute the transform from the base to node i
            g = np.matmul(g, utils.MatExp_se3([h*u1[i], h*u2[i], h*u3[i], h*v1[i], h*v2[i], h*v3[i]]))
            
        x_dir = g[0:3, 0]  # tip transform x-direction (a coord is in this direction)
        y_dir = g[0:3, 1]  # tip transform y-direction (b coord is in this direction)
        tip_pos = g[0:3, 3]
        
        total_moment = np.zeros((1,3))
        for tip_force in applied_tip_forces:
            p = tip_pos + x_dir * tip_force.location[0] + y_dir * tip_force.location[1] # position at end of rod where force is applied

            if tip_force.dead_load:
                force = tip_force.force
            else:
                R = g[0:3, 0:3]
                force = np.matmul(R, tip_force.force)   # rotate force from body coords into global frame

            print(f"force position: {p}\t applied force:{force}\tdot product: {np.dot(force,p)}")
            # print(np.dot(force,p))
            # subtract the applied force from the total internal energy
            energy -= np.dot(force, p)

            total_moment += np.cross(p, force)

        print(f"total_moment: {total_moment}")
            
        return energy.item() # use .item() to turn the 1x1 matrix into a scalar

    def _totalEnergyWithMoments(self, Z, applied_tip_forces, applied_tip_moments):
        h = self.L / (self.n - 1)

        energy = 0
        g = np.eye(4) # current transform along the rod g(s)

        # extract variables
        a = Z[0:self.n]
        b = Z[self.n:2*self.n]
        c = Z[2*self.n:3*self.n]
        v1 = Z[3*self.n:4*self.n-1]
        v2 = Z[4*self.n-1:5*self.n-2]
        v3 = Z[5*self.n-2:6*self.n-3]
        u1 = Z[6*self.n-3:7*self.n-4]
        u2 = Z[7*self.n-4:8*self.n-5]
        u3 = Z[8*self.n-5:9*self.n-6]

        # iterate through the links (i.e. midpoints between the nodes)
        for i in range(0,self.n-1):
            # get cross-sectional parameters at midpoint between nodes
            a_mid = (a[i] + a[i+1])/2
            b_mid = (b[i] + b[i+1])/2
            c_mid = (c[i] + c[i+1])/2

            # get approximate derivates of a, b, and c
            a_prime = (a[i+1] - a[i])/h
            b_prime = (b[i+1] - b[i])/h
            c_prime = (c[i+1] - c[i])/h

            # compute energy for this segment
            k11 = self.E*(self.cross_section.Ix*b_mid**2 + self.cross_section.Iy*c_mid**2)
            k22 = self.E*(self.cross_section.Iy*a_mid**2 + self.cross_section.Ix*c_mid**2)
            k12 = -self.E*(self.cross_section.Iy*a_mid*c_mid + self.cross_section.Ix*b_mid*c_mid)
            k33 = self.G*self.cross_section.Iy*(a_mid**2 + c_mid**2) + self.G*self.cross_section.Ix*(b_mid**2 + c_mid**2)
            K_mat = np.matrix([ [k11, k12, 0], [k12, k22, 0], [0, 0, k33] ])
            u_vec = np.array([u1[i], u2[i], u3[i]])
            strain_vec = np.array([a_mid - 1, b_mid -1, v3[i] - 1])
            M_mat = self.K[0:3, 0:3]
            
            energy += 0.5 * h * (self.G*self.cross_section.A0*v1[i]**2 + self.G*self.cross_section.A0*v2[i]**2 + 4*self.G*self.cross_section.A0*c_mid**2 +
                                 np.matmul(np.matmul(u_vec, K_mat), u_vec) +
                                 self.cross_section.A0 * np.matmul(np.matmul(strain_vec, M_mat), strain_vec) +
                                 self.G*self.cross_section.Iy*a_prime**2 + self.G*self.cross_section.Ix*b_prime**2 + self.G*(self.cross_section.Ix + self.cross_section.Iy)*c_prime**2 +
                                 2*self.G*u3[i]*( (a_mid*c_prime - a_prime*c_mid)*self.cross_section.Iy - (b_mid*c_prime - b_prime*c_mid)*self.cross_section.Ix)
                                 )
            # compute energy stored in this segment
            # energy += 0.5 * h * (e0_energy + ex_energy + ey_energy)
            
            # compute the transform from the base to node i
            g = np.matmul(g, utils.MatExp_se3([h*u1[i], h*u2[i], h*u3[i], h*v1[i], h*v2[i], h*v3[i]]))
            
        x_dir = g[0:3, 0]  # tip transform x-direction (a coord is in this direction)
        y_dir = g[0:3, 1]  # tip transform y-direction (b coord is in this direction)
        tip_pos = g[0:3, 3]
        
        total_moment = np.zeros((1,3))
        for tip_force in applied_tip_forces:
            p = tip_pos + x_dir * tip_force.location[0] + y_dir * tip_force.location[1] # position at end of rod where force is applied

            if tip_force.dead_load:
                force = tip_force.force
            else:
                R = g[0:3, 0:3]
                force = np.matmul(R, tip_force.force)   # rotate force from body coords into global frame

            print(f"force position: {p}\t applied force:{force}\tdot product: {np.dot(force,p)}")
            # print(np.dot(force,p))
            # subtract the applied force from the total internal energy
            energy -= np.dot(force, p)

            total_moment += np.cross(p, force)

        for tip_moment in applied_tip_moments:
            omega = utils.MatLog_SO3(g[0:3,0:3])
            energy -= np.dot(tip_moment.moment, omega)
            
        return energy.item() # use .item() to turn the 1x1 matrix into a scalar

    def _totalEnergyWithMomentsAndTorsionalCorrection(self, Z, applied_tip_forces, applied_tip_moments, torsional_correction):
        h = self.L / (self.n - 1)

        energy = 0
        g = np.eye(4) # current transform along the rod g(s)

        # extract variables
        a = Z[0:self.n]
        b = Z[self.n:2*self.n]
        c = Z[2*self.n:3*self.n]
        v1 = Z[3*self.n:4*self.n-1]
        v2 = Z[4*self.n-1:5*self.n-2]
        v3 = Z[5*self.n-2:6*self.n-3]
        u1 = Z[6*self.n-3:7*self.n-4]
        u2 = Z[7*self.n-4:8*self.n-5]
        u3 = Z[8*self.n-5:9*self.n-6]

        # iterate through the links (i.e. midpoints between the nodes)
        for i in range(0,self.n-1):
            # get cross-sectional parameters at midpoint between nodes
            a_mid = (a[i] + a[i+1])/2
            b_mid = (b[i] + b[i+1])/2
            c_mid = (c[i] + c[i+1])/2

            # get approximate derivates of a, b, and c
            a_prime = (a[i+1] - a[i])/h
            b_prime = (b[i+1] - b[i])/h
            c_prime = (c[i+1] - c[i])/h

            # compute energy for this segment

            k11 = self.E*(self.cross_section.Ix*b_mid**2 + self.cross_section.Iy*c_mid**2)
            k22 = self.E*(self.cross_section.Iy*a_mid**2 + self.cross_section.Ix*c_mid**2)
            k12 = -self.E*(self.cross_section.Iy*a_mid*c_mid + self.cross_section.Ix*b_mid*c_mid)
            k33 = torsional_correction*self.G*self.cross_section.Iy*a_mid**2 + torsional_correction*self.G*self.cross_section.Ix*b_mid**2 + torsional_correction*self.G*(self.cross_section.Ix + self.cross_section.Iy)*c_mid**2
            K_mat = np.matrix([ [k11, k12, 0], [k12, k22, 0], [0, 0, k33] ])
            u_vec = np.array([u1[i], u2[i], u3[i]])
            strain_vec = np.array([a_mid - 1, b_mid -1, v3[i] - 1])
            M_mat = self.K[0:3, 0:3]
            
            energy += 0.5 * h * (self.G*self.cross_section.A0*v1[i]**2 + self.G*self.cross_section.A0*v2[i]**2 + 4*self.G*self.cross_section.A0*c_mid**2 +
                                 np.matmul(np.matmul(u_vec, K_mat), u_vec) +
                                 self.cross_section.A0 * np.matmul(np.matmul(strain_vec, M_mat), strain_vec) +
                                 self.G*self.cross_section.Iy*a_prime**2 + self.G*self.cross_section.Ix*b_prime**2 + self.G*torsional_correction*(self.cross_section.Ix + self.cross_section.Iy)*c_prime**2 +
                                 2*self.G*u3[i]*( (a_mid*c_prime - a_prime*c_mid)*self.cross_section.Iy - (b_mid*c_prime - b_prime*c_mid)*self.cross_section.Ix)
                                 )
            # compute energy stored in this segment
            # energy += 0.5 * h * (e0_energy + ex_energy + ey_energy)
            
            # compute the transform from the base to node i
            g = np.matmul(g, utils.MatExp_se3([h*u1[i], h*u2[i], h*u3[i], h*v1[i], h*v2[i], h*v3[i]]))
            
        x_dir = g[0:3, 0]  # tip transform x-direction (a coord is in this direction)
        y_dir = g[0:3, 1]  # tip transform y-direction (b coord is in this direction)
        tip_pos = g[0:3, 3]
        
        total_moment = np.zeros((1,3))
        for tip_force in applied_tip_forces:
            p = tip_pos + x_dir * tip_force.location[0] + y_dir * tip_force.location[1] # position at end of rod where force is applied

            if tip_force.dead_load:
                force = tip_force.force
            else:
                R = g[0:3, 0:3]
                force = np.matmul(R, tip_force.force)   # rotate force from body coords into global frame

            print(f"force position: {p}\t applied force:{force}\tdot product: {np.dot(force,p)}")
            # print(np.dot(force,p))
            # subtract the applied force from the total internal energy
            energy -= np.dot(force, p)

            total_moment += np.cross(p, force)

        for tip_moment in applied_tip_moments:
            omega = utils.MatLog_SO3(g[0:3,0:3])
            energy -= np.dot(tip_moment.moment, omega)
            
        return energy.item() # use .item() to turn the 1x1 matrix into a scalar

    def _totalEnergyTorsionalCorrection(self, Z, applied_tip_forces):
        h = self.L / (self.n - 1)

        energy = 0
        g = np.eye(4) # current transform along the rod g(s)

        # extract variables
        a = Z[0:self.n]
        b = Z[self.n:2*self.n]
        c = Z[2*self.n:3*self.n]
        v1 = Z[3*self.n:4*self.n-1]
        v2 = Z[4*self.n-1:5*self.n-2]
        v3 = Z[5*self.n-2:6*self.n-3]
        u1 = Z[6*self.n-3:7*self.n-4]
        u2 = Z[7*self.n-4:8*self.n-5]
        u3 = Z[8*self.n-5:9*self.n-6]

        # iterate through the links (i.e. midpoints between the nodes)
        for i in range(0,self.n-1):
            # get cross-sectional parameters at midpoint between nodes
            a_mid = (a[i] + a[i+1])/2
            b_mid = (b[i] + b[i+1])/2
            c_mid = (c[i] + c[i+1])/2

            # get approximate derivates of a, b, and c
            a_prime = (a[i+1] - a[i])/h
            b_prime = (b[i+1] - b[i])/h
            c_prime = (c[i+1] - c[i])/h

            # compute energy for this segment
            torsional_correction = 0.141 / (1/6)    # hard-coded for a square right now

            k11 = self.E*(self.cross_section.Ix*b_mid**2 + self.cross_section.Iy*c_mid**2)
            k22 = self.E*(self.cross_section.Iy*a_mid**2 + self.cross_section.Ix*c_mid**2)
            k12 = -self.E*(self.cross_section.Iy*a_mid*c_mid + self.cross_section.Ix*b_mid*c_mid)
            k33 = self.G*self.cross_section.Iy*a_mid**2 + self.G*self.cross_section.Ix*b_mid**2 + torsional_correction*self.G*(self.cross_section.Ix + self.cross_section.Iy)*c_mid**2
            K_mat = np.matrix([ [k11, k12, 0], [k12, k22, 0], [0, 0, k33] ])
            u_vec = np.array([u1[i], u2[i], u3[i]])
            strain_vec = np.array([a_mid - 1, b_mid -1, v3[i] - 1])
            M_mat = self.K[0:3, 0:3]
            
            energy += 0.5 * h * (self.G*self.cross_section.A0*v1[i]**2 + self.G*self.cross_section.A0*v2[i]**2 + 4*self.G*self.cross_section.A0*c_mid**2 +
                                 np.matmul(np.matmul(u_vec, K_mat), u_vec) +
                                 self.cross_section.A0 * np.matmul(np.matmul(strain_vec, M_mat), strain_vec) +
                                 self.G*self.cross_section.Iy*a_prime**2 + self.G*self.cross_section.Ix*b_prime**2 + self.G*torsional_correction*(self.cross_section.Ix + self.cross_section.Iy)*c_prime**2 +
                                 2*self.G*u3[i]*( (a_mid*c_prime - a_prime*c_mid)*self.cross_section.Iy - (b_mid*c_prime - b_prime*c_mid)*self.cross_section.Ix)
                                 )
            # compute energy stored in this segment
            # energy += 0.5 * h * (e0_energy + ex_energy + ey_energy)
            
            # compute the transform from the base to node i
            g = np.matmul(g, utils.MatExp_se3([h*u1[i], h*u2[i], h*u3[i], h*v1[i], h*v2[i], h*v3[i]]))
            
        x_dir = g[0:3, 0]  # tip transform x-direction (a coord is in this direction)
        y_dir = g[0:3, 1]  # tip transform y-direction (b coord is in this direction)
        tip_pos = g[0:3, 3]
        
        total_moment = np.zeros((1,3))
        for tip_force in applied_tip_forces:
            p = tip_pos + x_dir * tip_force.location[0] + y_dir * tip_force.location[1] # position at end of rod where force is applied

            if tip_force.dead_load:
                force = tip_force.force
            else:
                R = g[0:3, 0:3]
                force = np.matmul(R, tip_force.force)   # rotate force from body coords into global frame

            # print(f"force position: {p}\t applied force:{force}\tdot product: {np.dot(force,p)}")
            # print(np.dot(force,p))
            # subtract the applied force from the total internal energy
            energy -= np.dot(force, p)

            total_moment += np.cross(p, force)

        # print(f"total_moment: {total_moment}")
        # print(f"energy: {energy.item()}")
        # print(f"tip position: ({tip_pos[0]}, {tip_pos[1]}, {tip_pos[2]})")
            
        return energy.item() # use .item() to turn the 1x1 matrix into a scalar


    # returns a list of 4x4 transformation matrices (starting with node 0, i.e. the base node) from the base to each node
    def nodeTransforms(self, node_start=0, node_start_pos=[0,0,0], node_start_orientation=np.eye(3)):
        h = self.L / (self.n-1)

        # extract variables
        v1 = self.Z[3*self.n:4*self.n-1]
        v2 = self.Z[4*self.n-1:5*self.n-2]
        v3 = self.Z[5*self.n-2:6*self.n-3]
        u1 = self.Z[6*self.n-3:7*self.n-4]
        u2 = self.Z[7*self.n-4:8*self.n-5]
        u3 = self.Z[8*self.n-5:9*self.n-6]

        # create vertices for undeformed cross-section
        xsection_points = self.cross_section.meshPoints()
        _, num_xs_points = xsection_points.shape

        # create matrix for all vertices in the mesh
        vertices = np.zeros((3, num_xs_points*self.n))

        transforms = [np.eye(4) for i in range(self.n)]

        g = np.eye(4)
        g[0:3,3] = node_start_pos
        g[0:3,0:3] = node_start_orientation
        transforms[node_start] = g
        for i in range(node_start,self.n-1):
            # compute the transform from the base to node i+1
            g = np.matmul(g, utils.MatExp_se3([h*u1[i], h*u2[i], h*u3[i], h*v1[i], h*v2[i], h*v3[i]]))

            transforms[i+1] = g
        
        g = transforms[node_start]
        for i in range(node_start-1,-1,-1):
            g = np.matmul(g, utils.MatExp_se3([-h*u1[i], -h*u2[i], -h*u3[i], -h*v1[i], -h*v2[i], -h*v3[i]]))

            transforms[i] = g
        
        return transforms

    def asMesh(self, node_start=0, node_start_pos=[0,0,0], node_start_orientation=np.eye(3)):
        h = self.L / (self.n-1)

        # extract variables
        a = self.Z[0:self.n]
        b = self.Z[self.n:2*self.n]
        c = self.Z[2*self.n:3*self.n]

        node_xsections = self.nodeCrossSectionPolyData(node_start, node_start_pos, node_start_orientation)
        num_xs_points = node_xsections[0].points.shape[0] # number of points in each cross section

         # create matrix for all vertices in the mesh
        vertices = np.zeros((num_xs_points*self.n, 3))

        for i,xsection in enumerate(node_xsections):
            # add the points to the overall matrix
            vertices[num_xs_points*(i):num_xs_points*(i+1),:] = xsection.points

        # create 'side' faces for mesh between each cross section
        faces = np.zeros((num_xs_points*(self.n-1), 5), dtype=int)
        for i in range(self.n-1):
            for k in range(num_xs_points):
                p1 = i*num_xs_points + k
                p2 = i*num_xs_points + (k+1)
                p3 = (i+1)*num_xs_points + (k+1)
                p4 = (i+1)*num_xs_points + k
                if (k == num_xs_points - 1):
                    p2 = i*num_xs_points
                    p3 = (i+1)*num_xs_points
                
                faces[i*num_xs_points + k, :] = [4, p1, p2, p3, p4]
        
        # create end faces
        # s=0 end face
        bottom_face = list(range(num_xs_points))
        bottom_face.insert(0, len(bottom_face))

        # s=L end face
        top_face = list(range(num_xs_points*(self.n-1), num_xs_points*self.n))
        top_face.insert(0, len(top_face))
        
        # concatenate faces into one array - PyVista reads them in as a flat array
        all_faces = np.hstack([faces.flatten(), bottom_face, top_face])
        
        # create and plot the mesh with PyVista
        surf = pv.PolyData(vertices, all_faces)
        surf.triangulate(inplace=True)
        return surf

    # returns the cross sections of the rod at each node along the rod as a pv.PolyData object
    def nodeCrossSectionPolyData(self, node_start=0, node_start_pos=[0,0,0], node_orientation=np.eye(3), scale_factor=1.0):

        # get transforms from base to each node
        node_transforms = self.nodeTransforms(node_start, node_start_pos, node_orientation)

        # create vertices for undeformed cross-section
        xsection_points = self.cross_section.meshPoints()
        for p in xsection_points:
            p *= scale_factor

        _, num_xs_points = xsection_points.shape

        # list for all the poly data
        nodes_poly_data = []

        for i,transform in enumerate(node_transforms):
            R = transform[0:3, 0:3]
            p = transform[0:3, 3]

            C = self.nodeDistortionMatrix(i)
            # deform and transform the base circle into the current frame
            cur_xsection_points = np.matmul(R, np.matmul(C, xsection_points)) + p[:, np.newaxis] # use np.newaxis to explicitly create a column vector
            face = list(range(num_xs_points))
            face.insert(0,num_xs_points)

            poly_data = pv.PolyData(cur_xsection_points.transpose(), face)
            nodes_poly_data.append(poly_data)

        return nodes_poly_data

    def nodeCrossSectionPolyData2D(self):
        xsection_points = self.cross_section.meshPoints()
        _, num_xs_points = xsection_points.shape

        nodes_poly_data_2d = []
        for i in range(self.n):
            C = self.nodeDistortionMatrix(i)
            deformed_xsection_points = np.matmul(C, xsection_points)
            face = list(range(num_xs_points))
            face.insert(0, num_xs_points)

            poly_data = pv.PolyData(deformed_xsection_points.transpose(), face)
            nodes_poly_data_2d.append(poly_data)

        return nodes_poly_data_2d

    # returns the distortion matrix C for node i
    def nodeDistortionMatrix(self, i):
        # extract variables
        a = self.Z[0:self.n]
        b = self.Z[self.n:2*self.n]
        c = self.Z[2*self.n:3*self.n]

        C = np.array([[a[i], c[i], 0], [c[i], b[i], 0], [0, 0, 1]])

        return C
    
    def tipPosition(self, ab_coords=[0,0]):
        end_transform = self.nodeTransforms()[-1]
        p = end_transform[0:3, 3]

        C = self.nodeDistortionMatrix(-1)
        deformed_ab_coords = np.matmul(C[0:2,0:2], np.array(ab_coords))
        x_dir = end_transform[0:3,0]
        y_dir = end_transform[0:3,1]
        return p + x_dir * deformed_ab_coords[0] * self.cross_section.rx  + y_dir * deformed_ab_coords[1] * self.cross_section.ry

    ## given: an array of (s, a, b) coordinates
    # where s: [0,1] ; 0 is base, 1 is tip
    #       x: x-coordinate in cross-section plane
    #       y: y-coordinate in cross-section plane
    def positionsFromCosseratCoords(self, cosserat_coords):
        node_transforms = self.nodeTransforms()

        positions = np.zeros( (len(cosserat_coords), 3) )

        for i, (s,x,y) in enumerate(cosserat_coords):
            below_node = int(np.floor(s*(self.n-1)))
            if below_node == self.n-1:
                above_node = below_node
                interp_factor = 0
            else:
                above_node = below_node + 1
                interp_factor = s*(self.n-1) - below_node

            below_transform = node_transforms[below_node]
            above_transform = node_transforms[above_node]

            # just do linear interpolation (as a first pass)
            p = (1-interp_factor)*below_transform[0:3, 3] + interp_factor*above_transform[0:3, 3]
            R = (1-interp_factor)*below_transform[0:3, 0:3] + interp_factor*above_transform[0:3, 0:3]
            C = (1-interp_factor)*self.nodeDistortionMatrix(below_node) + interp_factor*self.nodeDistortionMatrix(above_node)
            r = np.array([x, y, 0])

            positions[i] = p + np.matmul(np.matmul(R, C), r)
        
        return positions




##############################################################################
# Cosserat rod with cross-sectional deformation that is linear in x and y
#  i.e. a(x,y,s) = a0 + ax * x + ay * y
#       b(x,y,s) = b0 + bx * x + by * y
#       c(x,y,s) = c0 + cx * x + cy * y
#
# Able to capture "trapezoidal" deformation mode during bending
#############################################################################
class LinearDeformationCosseratRod:
    def __init__(self, n, L, cross_section, E, nu):
        self.n = n
        self.L = L
        self.cross_section = cross_section

        # set material properties
        self.E = E
        self.nu = nu

        M = E * (1-nu) / ( (1+nu) * (1-2*nu))
        lam = E * nu / ( (1+nu) * (1-2*nu))
        self.G = E / (2 * (1+nu))       # shear modulus
        self.K = np.matrix([[ M, lam, lam, 0, 0, 0 ],   # stiffness matrix for Cosserat rod
                            [ lam, M, lam, 0, 0, 0 ],
                            [ lam, lam, M, 0, 0, 0 ],
                            [ 0, 0, 0, self.G, 0, 0 ],
                            [ 0, 0, 0, 0, self.G, 0 ],
                            [ 0, 0, 0, 0, 0, self.G ]])
        
        # initialize Z
        # define initial variables for optimization problem (variables in the undeformed state)
        #_ cross-sectional paramaters a, b, c are defined at each of the n nodes along the length
        a0_0 = np.ones(n)     # a0 = 1 when cross-section is undeformed
        ax_0 = np.zeros(n)    # ax = 0 when cross-section is undeformed
        ay_0 = np.zeros(n)    # ay = 0 when cross-section is undeformed
        b0_0 = np.ones(n)     # b0 = 1 when cross-section is undeformed
        bx_0 = np.zeros(n)    # bx = 0 when cross-section is undeformed
        by_0 = np.zeros(n)    # by = 0 when cross-section is undeformed
        c0_0 = np.zeros(n)    # c0 = 0 when cross-section is undeformed
        cx_0 = np.zeros(n)    # cx = 0 when cross-section is undeformed
        cy_0 = np.zeros(n)    # cy = 0 when cross-section is undeformed

        # link parameters are defined at the midpoint between the nodes, i.e. there are n-1 links
        v1_0 = np.zeros(n-1)  # v1 = 0 when no loads on rod
        v2_0 = np.zeros(n-1)  # v2 = 0 when no loads on rod
        v3_0 = np.ones(n-1)   # v3 = 1 when no loads on rod
        u1_0 = np.zeros(n-1)  # u1 = 0 when no loads on rod
        u2_0 = np.zeros(n-1)  # u2 = 0 when no loads on rod
        u3_0 = np.zeros(n-1)  # u3 = 0 when no loads on rod

        self.Z = np.hstack([a0_0, ax_0, ay_0, b0_0, bx_0, by_0, c0_0, cx_0, cy_0, v1_0, v2_0, v3_0, u1_0, u2_0, u3_0])

    def solveOptimizationProblem(self, applied_tip_forces):
        # put some bounds/constraints on Z
        # bounds of (None,None) indicate that a variable has no bounds in the optimization
        Z_bounds = [(None,None)]*self.Z.size # array of None's the size of Z

        # constrain cross-section at base of rod (s=0) to be undeformed
        Z_bounds[0] = (1,1)         # a0 at s=0 is 1 (undeformed)
        Z_bounds[self.n] = (0,0)    # ax at s=0 is 0 (undeformed)
        Z_bounds[2*self.n] = (0,0)  # ay at s=0 is 0 (undeformed)
        Z_bounds[3*self.n] = (1,1)  # b0 at s=0 is 1 (undeformed)
        Z_bounds[4*self.n] = (0,0)  # bx at s=0 is 0 (undeformed)
        Z_bounds[5*self.n] = (0,0)  # by at s=0 is 0 (undeformed)
        Z_bounds[6*self.n] = (0,0)  # c0 at s=0 is 0 (undeformed)
        Z_bounds[7*self.n] = (0,0)  # cx at s=0 is 0 (undeformed)
        Z_bounds[8*self.n] = (0,0)  # cy at s=0 is 0 (undeformed)

        # constrain cross-section at end of rod (s=L) to be undeformed
        Z_bounds[self.n-1] = (1,1)    # a0 at s=L is 1 (undeformed)
        Z_bounds[2*self.n-1] = (0,0)  # ax at s=L is 0 (undeformed)
        Z_bounds[3*self.n-1] = (0,0)  # ay at s=L is 0 (undeformed)
        Z_bounds[4*self.n-1] = (1,1)  # b0 at s=L is 1 (undeformed)
        Z_bounds[5*self.n-1] = (0,0)  # bx at s=L is 0 (undeformed)
        Z_bounds[6*self.n-1] = (0,0)  # by at s=L is 0 (undeformed)
        Z_bounds[7*self.n-1] = (0,0)  # c0 at s=L is 0 (undeformed)
        Z_bounds[8*self.n-1] = (0,0)  # cx at s=L is 0 (undeformed)
        Z_bounds[9*self.n-1] = (0,0)  # cy at s=L is 0 (undeformed)

        # constrain the rest of a, b, and c to be reasonable values (shouldn't be needed)
        # for i in range(1,self.n-1):
        #     Z_bounds[i] = (0, 10)
        #     Z_bounds[self.n + i] = (0, 10)
        #     Z_bounds[2*self.n + i] = (-10, 10)

        # constrain v1, v2, v3 to be reasonable values (shouldn't be needed)
        # for i in range(0, n-1):
        #     Z_bounds[3*n + i] = (0,0)
        #     Z_bounds[3*n + n-1 + i] = (0,0)
        #     Z_bounds[3*n + 2*(n-1) + i] = (0.5,2)

        # constrain u1, u2, u3 to be reasonable values (shouldn't be needed)
        # for i in range(0, n-1):
        #     Z_bounds[3*n + 3*(n-1) + i] = (-1,1)
        #     Z_bounds[3*n + 4*(n-1) + i] = (-1,1)
        #     Z_bounds[3*n + 5*(n-1) + i] = (-1,1)

        res = opt.minimize(self._totalEnergy, self.Z, args=(applied_tip_forces), method='slsqp', bounds=Z_bounds, options={'maxiter': 10000, 'disp':True})
        self.Z = res.x
    
        a0 = self.Z[0:self.n]
        ax = self.Z[self.n:2*self.n]
        ay = self.Z[2*self.n:3*self.n]

        b0 = self.Z[3*self.n:4*self.n]
        bx = self.Z[4*self.n:5*self.n]
        by = self.Z[5*self.n:6*self.n]

        c0 = self.Z[6*self.n:7*self.n]
        cx = self.Z[7*self.n:8*self.n]
        cy = self.Z[8*self.n:9*self.n]
        
        v1 = self.Z[9*self.n:10*self.n-1]
        v2 = self.Z[10*self.n-1:11*self.n-2]
        v3 = self.Z[11*self.n-2:12*self.n-3]
        u1 = self.Z[12*self.n-3:13*self.n-4]
        u2 = self.Z[13*self.n-4:14*self.n-5]
        u3 = self.Z[14*self.n-5:15*self.n-6]

        print(f"a0: {a0}")
        print(f"ax: {ax}")
        print(f"ay: {ay}")
        print(f"b0: {b0}")
        print(f"bx: {bx}")
        print(f"by: {by}")
        print(f"c0: {c0}")
        print(f"cx: {cx}")
        print(f"cy: {cy}")
    
    def _totalEnergy(self, Z, applied_tip_forces):
        h = self.L / (self.n - 1)

        energy = 0
        g = np.eye(4) # current transform along the rod g(s)

        # extract variables
        a0 = Z[0:self.n]
        ax = Z[self.n:2*self.n]
        ay = Z[2*self.n:3*self.n]

        b0 = Z[3*self.n:4*self.n]
        bx = Z[4*self.n:5*self.n]
        by = Z[5*self.n:6*self.n]

        c0 = Z[6*self.n:7*self.n]
        cx = Z[7*self.n:8*self.n]
        cy = Z[8*self.n:9*self.n]
        
        v1 = Z[9*self.n:10*self.n-1]
        v2 = Z[10*self.n-1:11*self.n-2]
        v3 = Z[11*self.n-2:12*self.n-3]
        u1 = Z[12*self.n-3:13*self.n-4]
        u2 = Z[13*self.n-4:14*self.n-5]
        u3 = Z[14*self.n-5:15*self.n-6]

        # iterate through the links (i.e. midpoints between the nodes)
        for i in range(0,self.n-1):
            # get cross-sectional parameters at midpoint between nodes
            a0_mid = (a0[i] + a0[i+1])/2
            ax_mid = (ax[i] + ax[i+1])/2
            ay_mid = (ay[i] + ay[i+1])/2

            b0_mid = (b0[i] + b0[i+1])/2
            bx_mid = (bx[i] + bx[i+1])/2
            by_mid = (by[i] + by[i+1])/2

            c0_mid = (c0[i] + c0[i+1])/2
            cx_mid = (cx[i] + cx[i+1])/2
            cy_mid = (cy[i] + cy[i+1])/2

            # get approximate derivates of a, b, and c
            a0_prime = (a0[i+1] - a0[i])/h
            ax_prime = (ax[i+1] - ax[i])/h
            ay_prime = (ay[i+1] - ay[i])/h

            b0_prime = (b0[i+1] - b0[i])/h
            bx_prime = (bx[i+1] - bx[i])/h
            by_prime = (by[i+1] - by[i])/h

            c0_prime = (c0[i+1] - c0[i])/h
            cx_prime = (cx[i+1] - cx[i])/h
            cy_prime = (cy[i+1] - cy[i])/h

            # strain_0, strain_x, and strain_y (directly from the paper)

            e0 = np.array([ a0_mid - 1,
                            b0_mid - 1,
                            v3[i] - 1,
                            v2[i],
                            v1[i],
                            2*c0_mid ])
            
            ex = np.array([ 2*ax_mid,
                            bx_mid + cy_mid,
                            c0_mid * u1[i] - a0_mid * u2[i],
                            c0_prime + a0_mid * u3[i],
                            a0_prime - c0_mid * u3[i],
                            3*cx_mid + ay_mid ])
            
            ey = np.array([ cx_mid + ay_mid,
                            2*by_mid,
                            b0_mid * u1[i] - c0_mid * u2[i],
                            b0_prime + c0_mid*u3[i],
                            c0_prime - b0_mid*u3[i],
                            3*cy_mid + bx_mid ])
            
            exy = np.array([0,
                            0,
                            -ay_mid*u2[i] + cy_mid*u1[i] + bx_mid*u1[i] - cx_mid*u2[i],
                            bx_prime + cy_prime + ay_mid*u3[i] + cx_mid*u3[i],
                            ay_prime + cx_prime - bx_mid*u3[i] - cy_mid*u3[i],
                            0])
            
            ex2 = np.array([0,
                            0,
                            -ax_mid*u2[i] + cx_mid*u1[i],
                            cx_prime + ax_mid*u3[i],
                            ax_prime - cx_mid*u3[i],
                            0])
            
            ey2 = np.array([0,
                            0,
                            by_mid*u1[i] - cy_mid*u2[i],
                            by_prime + cy_mid*u3[i],
                            cy_prime - by_mid*u3[i],
                            0])

            # compute strain energy density for this segment
            e0_energy = self.cross_section.A0 * np.matmul(np.matmul(e0, self.K), e0)
            e0_ex2_energy = 2*self.cross_section.Iy * np.matmul(np.matmul(e0, self.K), ex2)
            e0_ey2_energy = 2*self.cross_section.Ix * np.matmul(np.matmul(e0, self.K), ey2)
            exy_energy = self.cross_section.Ix2y2 * np.matmul(np.matmul(exy, self.K), exy)
            ex2_ey2_energy = 2*self.cross_section.Ix2y2 * np.matmul(np.matmul(ex2, self.K), ey2)
            ex2_energy = self.cross_section.Iy2 * np.matmul(np.matmul(ex2, self.K), ex2)
            ey2_energy = self.cross_section.Ix2 * np.matmul(np.matmul(ey2, self.K), ey2)
            ex_energy = self.cross_section.Iy * np.matmul(np.matmul(ex, self.K), ex)
            ey_energy = self.cross_section.Ix * np.matmul(np.matmul(ey, self.K), ey)

            # compute energy stored in this segment
            energy += 0.5 * h * (e0_energy + e0_ex2_energy + e0_ey2_energy + ex_energy + ey_energy + exy_energy + ex2_ey2_energy + ex2_energy + ey2_energy)
            
            # compute the transform from the base to node i
            g = np.matmul(g, utils.MatExp_se3([h*u1[i], h*u2[i], h*u3[i], h*v1[i], h*v2[i], h*v3[i]]))
            
        x_dir = g[0:3, 0]  # tip transform x-direction (a coord is in this direction)
        y_dir = g[0:3, 1]  # tip transform y-direction (b coord is in this direction)
        tip_pos = g[0:3, 3]
        
        for tip_force in applied_tip_forces:
            p = tip_pos + x_dir * tip_force.location[0] + y_dir * tip_force.location[1] # position at end of rod where force is applied

            if tip_force.dead_load:
                force = tip_force.force
            else:
                R = g[0:3, 0:3]
                force = np.matmul(R, tip_force.force)   # rotate force from body coords into global frame

            # subtract the applied force from the total internal energy
            energy -= np.dot(force, p)
            
        return energy.item() # use .item() to turn the 1x1 matrix into a scalar

    # returns a list of 4x4 transformation matrices (starting with node 0, i.e. the base node) from the base to each node
    def nodeTransforms(self):
        h = self.L / (self.n-1)

        # extract variables
        v1 = self.Z[9*self.n:10*self.n-1]
        v2 = self.Z[10*self.n-1:11*self.n-2]
        v3 = self.Z[11*self.n-2:12*self.n-3]
        u1 = self.Z[12*self.n-3:13*self.n-4]
        u2 = self.Z[13*self.n-4:14*self.n-5]
        u3 = self.Z[14*self.n-5:15*self.n-6]

        # create vertices for undeformed cross-section
        xsection_points = self.cross_section.meshPoints()
        _, num_xs_points = xsection_points.shape

        # create matrix for all vertices in the mesh
        vertices = np.zeros((3, num_xs_points*self.n))

        transforms = []

        g = np.eye(4)
        transforms.append(g)
        for i in range(self.n-1):
            # extract rotation and position from transformation
            R = g[0:3, 0:3]
            p = g[0:3, 3]

            # compute the transform from the base to node i+1
            g = np.matmul(g, utils.MatExp_se3([h*u1[i], h*u2[i], h*u3[i], h*v1[i], h*v2[i], h*v3[i]]))

            transforms.append(g)
        
        return transforms

    def asMesh(self):
        node_xsections = self.nodeCrossSectionPolyData()
        num_xs_points = node_xsections[0].points.shape[0] # number of points in each cross section

         # create matrix for all vertices in the mesh
        vertices = np.zeros((num_xs_points*self.n, 3))

        for i,xsection in enumerate(node_xsections):
            # add the points to the overall matrix
            vertices[num_xs_points*(i):num_xs_points*(i+1),:] = xsection.points

        # create 'side' faces for mesh between each cross section
        faces = np.zeros((num_xs_points*(self.n-1), 5), dtype=int)
        for i in range(self.n-1):
            for k in range(num_xs_points):
                p1 = i*num_xs_points + k
                p2 = i*num_xs_points + (k+1)
                p3 = (i+1)*num_xs_points + (k+1)
                p4 = (i+1)*num_xs_points + k
                if (k == num_xs_points - 1):
                    p2 = i*num_xs_points
                    p3 = (i+1)*num_xs_points
                
                faces[i*num_xs_points + k, :] = [4, p1, p2, p3, p4]
        
        # create end faces
        # s=0 end face
        bottom_face = list(range(num_xs_points))
        bottom_face.insert(0, len(bottom_face))

        # s=L end face
        top_face = list(range(num_xs_points*(self.n-1), num_xs_points*self.n))
        top_face.insert(0, len(top_face))
        
        # concatenate faces into one array - PyVista reads them in as a flat array
        all_faces = np.hstack([faces.flatten(), bottom_face, top_face])
        
        # create and plot the mesh with PyVista
        surf = pv.PolyData(vertices, all_faces)
        surf.triangulate(inplace=True)
        return surf

    # returns the cross sections of the rod at each node along the rod as a pv.PolyData object
    def nodeCrossSectionPolyData(self, scale_factor=1.0):

        # get transforms from base to each node
        node_transforms = self.nodeTransforms()

        # create vertices for undeformed cross-section
        xsection_points = self.cross_section.meshPoints()
        for p in xsection_points:
            p *= scale_factor

        _, num_xs_points = xsection_points.shape

        # list for all the poly data
        nodes_poly_data = []

        for i,transform in enumerate(node_transforms):
            R = transform[0:3, 0:3]
            p = transform[0:3, 3]

            deformed_xsection_points = np.copy(xsection_points)
            for pi in range(num_xs_points):
                C = self.nodeDistortionMatrix(i, xsection_points[0,pi], xsection_points[1,pi])
                deformed_xsection_points[:,pi] = np.matmul(C, xsection_points[:,pi])

            # deform and transform the base circle into the current frame
            cur_xsection_points = np.matmul(R, deformed_xsection_points) + p[:, np.newaxis] # use np.newaxis to explicitly create a column vector
            face = list(range(num_xs_points))
            face.insert(0,num_xs_points)

            poly_data = pv.PolyData(cur_xsection_points.transpose(), face)
            nodes_poly_data.append(poly_data)

        return nodes_poly_data

    def nodeCrossSectionPolyData2D(self):
        xsection_points = self.cross_section.meshPoints()
        _, num_xs_points = xsection_points.shape

        nodes_poly_data_2d = []
        
        for i in range(self.n):
            deformed_xsection_points = xsection_points
            for pi in range(num_xs_points):
                C = self.nodeDistortionMatrix(i, xsection_points[0,pi], xsection_points[1,pi])
                deformed_xsection_points[:,pi] = np.matmul(C, xsection_points[:,pi])

            face = list(range(num_xs_points))
            face.insert(0, num_xs_points)

            poly_data = pv.PolyData(deformed_xsection_points.transpose(), face)
            nodes_poly_data_2d.append(poly_data)

        return nodes_poly_data_2d

    # returns the distortion matrix C for node i
    def nodeDistortionMatrix(self, i, x, y):
        # extract variables
        a0 = self.Z[0:self.n]
        ax = self.Z[self.n:2*self.n]
        ay = self.Z[2*self.n:3*self.n]

        b0 = self.Z[3*self.n:4*self.n]
        bx = self.Z[4*self.n:5*self.n]
        by = self.Z[5*self.n:6*self.n]

        c0 = self.Z[6*self.n:7*self.n]
        cx = self.Z[7*self.n:8*self.n]
        cy = self.Z[8*self.n:9*self.n]

        a = a0[i] + ax[i]*x + ay[i]*y
        b = b0[i] + bx[i]*x + by[i]*y
        c = c0[i] + cx[i]*x + cy[i]*y
        C = np.array([[a, c, 0], [c, b, 0], [0, 0, 1]])

        return C
    
    def tipPosition(self, ab_coords=[0,0]):
        end_transform = self.nodeTransforms()[-1]
        p = end_transform[0:3, 3]
        return p
        # C = self.nodeDistortionMatrix(-1)
        # deformed_ab_coords = np.matmul(C[0:2,0:2], np.array(ab_coords))
        # x_dir = end_transform[0:3,0]
        # y_dir = end_transform[0:3,1]
        # return p + x_dir * deformed_ab_coords[0] * self.cross_section.rx  + y_dir * deformed_ab_coords[1] * self.cross_section.ry


##############################################################################
# Cosserat rod with internal hollow pressurized chamber
##############################################################################
class PressurizedCosseratRod:
    def __init__(self, n, L, cross_section, E, nu):
        self.n = n
        self.L = L
        self.cross_section = cross_section

        # set material properties
        self.E = E
        self.nu = nu

        M = E * (1-nu) / ( (1+nu) * (1-2*nu))
        lam = E * nu / ( (1+nu) * (1-2*nu))
        self.G = E / (2 * (1+nu))       # shear modulus
        self.K = np.matrix([[ M, lam, lam, 0, 0, 0 ],   # stiffness matrix for Cosserat rod
                            [ lam, M, lam, 0, 0, 0 ],
                            [ lam, lam, M, 0, 0, 0 ],
                            [ 0, 0, 0, self.G, 0, 0 ],
                            [ 0, 0, 0, 0, self.G, 0 ],
                            [ 0, 0, 0, 0, 0, self.G ]])
        
        # initialize Z
        # define initial variables for optimization problem (variables in the undeformed state)
        #_ cross-sectional paramaters a, b, c are defined at each of the n nodes along the length
        a_0 = np.ones(n)      # a = 1 when cross-section is undeformed
        b_0 = np.ones(n)      # b = 1 when cross-section is undeformed
        c_0 = np.zeros(n)     # c = 0 when cross-section is undeformed

        # link parameters are defined at the midpoint between the nodes, i.e. there are n-1 links
        v1_0 = np.zeros(n-1)  # v1 = 0 when no loads on rod
        v2_0 = np.zeros(n-1)  # v2 = 0 when no loads on rod
        v3_0 = np.ones(n-1)   # v3 = 1 when no loads on rod
        u1_0 = np.zeros(n-1)  # u1 = 0 when no loads on rod
        u2_0 = np.zeros(n-1)  # u2 = 0 when no loads on rod
        u3_0 = np.zeros(n-1)  # u3 = 0 when no loads on rod

        self.Z = np.hstack([a_0, b_0, c_0, v1_0, v2_0, v3_0, u1_0, u2_0, u3_0])

    def solveOptimizationProblem(self, applied_tip_forces, chamber_pressure):
        # put some bounds/constraints on Z
        # bounds of (None,None) indicate that a variable has no bounds in the optimization
        Z_bounds = [(None,None)]*self.Z.size # array of None's the size of Z

        # constrain cross-section at base of rod (s=0) to be undeformed
        Z_bounds[0] = (1,1)     # a at s=0 is 1 (undeformed)
        Z_bounds[self.n] = (1,1)     # b at s=0 is 1 (undeformed)
        Z_bounds[2*self.n] = (0,0)   # c at s=0 is 0 (undeformed)

        # constrain cross-section at end of rod (s=L) to be undeformed
        Z_bounds[self.n-1] = (1,1)   # a at s=L is 1 (undeformed)
        Z_bounds[2*self.n-1] = (1,1) # b at s=L is 1 (undeformed)
        Z_bounds[3*self.n-1] = (0,0) # c at s=L is 0 (undeformed)

        # constrain the rest of a, b, and c to be reasonable values (shouldn't be needed)
        for i in range(1,self.n-1):
            Z_bounds[i] = (0, 2)
            Z_bounds[self.n + i] = (0, 2)
            Z_bounds[2*self.n + i] = (-2, 2)

        # constrain v1, v2, v3 to be reasonable values (shouldn't be needed)
        # for i in range(0, self.n-1):
        #     Z_bounds[3*self.n + i] = (0,0)
        #     Z_bounds[3*self.n + self.n-1 + i] = (0,0)
        #     Z_bounds[3*self.n + 2*(self.n-1) + i] = (0.5,2)

        # constrain u1, u2, u3 to be reasonable values (shouldn't be needed)
        # for i in range(0, self.n-1):
        #     Z_bounds[3*self.n + 3*(self.n-1) + i] = (0,0)
        #     Z_bounds[3*self.n + 4*(self.n-1) + i] = (0,0)
        #     Z_bounds[3*self.n + 5*(self.n-1) + i] = (0,0)

        res = opt.minimize(self._totalEnergy, self.Z, args=(applied_tip_forces, chamber_pressure), method='slsqp', bounds=Z_bounds, options={'maxiter': 10000, 'disp':True})
        self.Z = res.x

        a =  self.Z[0:self.n]
        b =  self.Z[self.n:2*self.n]
        c =  self.Z[2*self.n:3*self.n]
        v1 = self.Z[3*self.n:4*self.n-1]
        v2 = self.Z[4*self.n-1:5*self.n-2]
        v3 = self.Z[5*self.n-2:6*self.n-3]
        u1 = self.Z[6*self.n-3:7*self.n-4]
        u2 = self.Z[7*self.n-4:8*self.n-5]
        u3 = self.Z[8*self.n-5:9*self.n-6]

        print(f"a: {a}")
        print(f"b: {b}")
        print(f"c: {c}")

    def _totalEnergy(self, Z, applied_tip_forces, chamber_pressure):
        h = self.L / (self.n - 1)

        current_chamber_volume = 0
        current_robot_volume = 0
        energy = 0
        g = np.eye(4) # current transform along the rod g(s)

        # extract variables
        a = Z[0:self.n]
        b = Z[self.n:2*self.n]
        c = Z[2*self.n:3*self.n]
        v1 = Z[3*self.n:4*self.n-1]
        v2 = Z[4*self.n-1:5*self.n-2]
        v3 = Z[5*self.n-2:6*self.n-3]
        u1 = Z[6*self.n-3:7*self.n-4]
        u2 = Z[7*self.n-4:8*self.n-5]
        u3 = Z[8*self.n-5:9*self.n-6]

        # iterate through the links (i.e. midpoints between the nodes)
        for i in range(0,self.n-1):
            # get cross-sectional parameters at midpoint between nodes
            a_mid = (a[i] + a[i+1])/2
            b_mid = (b[i] + b[i+1])/2
            c_mid = (c[i] + c[i+1])/2

            # get approximate derivates of a, b, and c
            a_prime = (a[i+1] - a[i])/h
            b_prime = (b[i+1] - b[i])/h
            c_prime = (c[i+1] - c[i])/h

            # compute energy for this segment
            k11 = self.E*(self.cross_section.Ix*b_mid**2 + self.cross_section.Iy*c_mid**2)
            k22 = self.E*(self.cross_section.Iy*a_mid**2 + self.cross_section.Ix*c_mid**2)
            k12 = -self.E*(self.cross_section.Iy*a_mid*c_mid + self.cross_section.Ix*b_mid*c_mid)
            k33 = self.G*self.cross_section.Iy*(a_mid**2 + c_mid**2) + self.G*self.cross_section.Ix*(b_mid**2 + c_mid**2)
            K_mat = np.matrix([ [k11, k12, 0], [k12, k22, 0], [0, 0, k33] ])
            u_vec = np.array([u1[i], u2[i], u3[i]])
            strain_vec = np.array([a_mid - 1, b_mid -1, v3[i] - 1])
            M_mat = self.K[0:3, 0:3]
            energy += 0.5 * h * (self.G*self.cross_section.A0*v1[i]**2 + self.G*self.cross_section.A0*v2[i]**2 + 4*self.G*self.cross_section.A0*c_mid**2 +
                                 np.matmul(np.matmul(u_vec, K_mat), u_vec) +
                                 self.cross_section.A0 * np.matmul(np.matmul(strain_vec, M_mat), strain_vec) +
                                 self.G*self.cross_section.Iy*a_prime**2 + self.G*self.cross_section.Ix*b_prime**2 + self.G*(self.cross_section.Ix + self.cross_section.Iy)*c_prime**2 +
                                 2*self.G*u3[i]*( (a_mid*c_prime - a_prime*c_mid)*self.cross_section.Iy - (b_mid*c_prime - b_prime*c_mid)*self.cross_section.Ix)
                                 )
            # compute energy stored in this segment
            # energy += 0.5 * h * (e0_energy + ex_energy + ey_energy)
            
            # compute the transform from the base to node i
            g = np.matmul(g, utils.MatExp_se3([h*u1[i], h*u2[i], h*u3[i], h*v1[i], h*v2[i], h*v3[i]]))

            # compute the cavity volume contribution for this rod segment
            xa_real = self.cross_section.xa - self.cross_section.cx
            ya_real = self.cross_section.ya - self.cross_section.cy
            D_mat_lazy = np.matrix([ [a_mid, c_mid, 0], [c_mid, b_mid, 0], [0, 0, v3[i] - (a_mid*u2[i] - c_mid*u1[i])*xa_real + (b_mid*u1[i] - c_mid*u2[i])*ya_real]])
            V_chamber_segment = h * np.linalg.det(D_mat_lazy) * self.cross_section.xs_inner.A0
            V_robot_segment = h * np.linalg.det(D_mat_lazy) * self.cross_section.xs_outer.A0
            current_chamber_volume += V_chamber_segment
            current_robot_volume += V_robot_segment
            
        x_dir = g[0:3, 0]  # tip transform x-direction (a coord is in this direction)
        y_dir = g[0:3, 1]  # tip transform y-direction (b coord is in this direction)
        tip_pos = g[0:3, 3]
        
        for tip_force in applied_tip_forces:
            p = tip_pos + x_dir * tip_force.location[0] + y_dir * tip_force.location[1] # position at end of rod where force is applied

            if tip_force.dead_load:
                force = tip_force.force
            else:
                R = g[0:3, 0:3]
                force = np.matmul(R, tip_force.force)   # rotate force from body coords into global frame

            # subtract the applied force from the total internal energy
            energy -= np.dot(force, p)

        # compute the potential energy associated with pressure/volume change
        initial_chamber_volume = self.cross_section.xs_inner.A0 * self.L
        adiabatic_chamber_work = (chamber_pressure*current_chamber_volume - chamber_pressure*initial_chamber_volume)
        initial_robot_volume = self.cross_section.xs_outer.A0 * self.L
        adiabatic_robot_work = (101.325e3*initial_robot_volume - 101.325e3*current_robot_volume) / (1 - 1.4)
        print(f"Chamber volume change: {current_chamber_volume - initial_chamber_volume}\tChamber Work: {adiabatic_chamber_work}\tRobot Volume Change: {initial_robot_volume - current_robot_volume}\tRobot Work: {adiabatic_robot_work}")

        energy -= adiabatic_chamber_work

        return energy.item() # use .item() to turn the 1x1 matrix into a scalar


    # returns a list of 4x4 transformation matrices (starting with node 0, i.e. the base node) from the base to each node
    def nodeTransforms(self):
        h = self.L / (self.n-1)

        # extract variables
        v1 = self.Z[3*self.n:4*self.n-1]
        v2 = self.Z[4*self.n-1:5*self.n-2]
        v3 = self.Z[5*self.n-2:6*self.n-3]
        u1 = self.Z[6*self.n-3:7*self.n-4]
        u2 = self.Z[7*self.n-4:8*self.n-5]
        u3 = self.Z[8*self.n-5:9*self.n-6]

        transforms = []

        g = np.eye(4)
        transforms.append(g)
        for i in range(self.n-1):
            # extract rotation and position from transformation
            R = g[0:3, 0:3]
            p = g[0:3, 3]

            # compute the transform from the base to node i+1
            g = np.matmul(g, utils.MatExp_se3([h*u1[i], h*u2[i], h*u3[i], h*v1[i], h*v2[i], h*v3[i]]))

            transforms.append(g)
        
        return transforms

    def asMesh(self):
        h = self.L / (self.n-1)

        # extract variables
        a = self.Z[0:self.n]
        b = self.Z[self.n:2*self.n]
        c = self.Z[2*self.n:3*self.n]

        node_xsections = self.nodeCrossSectionPolyData()
        
        outer_xsection_points = self.cross_section.xs_outer.meshPoints()
        inner_xsection_points = self.cross_section.xs_inner.meshPoints()
        _, num_outer_xs_points = outer_xsection_points.shape
        _, num_inner_xs_points = inner_xsection_points.shape

         # create matrix for all vertices in the mesh
        vertices = np.zeros(((num_outer_xs_points + num_inner_xs_points)*self.n, 3))

        for i,xsection in enumerate(node_xsections):
            # add the points to the overall matrix
            vertices[(num_outer_xs_points + num_inner_xs_points)*(i):(num_outer_xs_points + num_inner_xs_points)*(i+1),:] = xsection.points

        # create 'side' faces in between outer cross section
        side_faces = np.zeros(((num_outer_xs_points + num_inner_xs_points)*(self.n-1), 5), dtype=int)
        for i in range(self.n-1):
            for k in range(num_outer_xs_points):
                p1 = i*(num_outer_xs_points + num_inner_xs_points) + k
                p2 = i*(num_outer_xs_points + num_inner_xs_points) + (k+1)
                p3 = (i+1)*(num_outer_xs_points + num_inner_xs_points) + (k+1)
                p4 = (i+1)*(num_outer_xs_points + num_inner_xs_points) + k

                if (k == num_outer_xs_points - 1):
                    p2 = i*(num_outer_xs_points + num_inner_xs_points)
                    p3 = (i+1)*(num_outer_xs_points + num_inner_xs_points)
                
                side_faces[i*(num_outer_xs_points + num_inner_xs_points) + k, :] = [4, p1, p2, p3, p4]

        # create 'side' faces in between inner cross section
        for i in range(self.n-1):
            for k in range(num_inner_xs_points):
                p1 = i*(num_outer_xs_points + num_inner_xs_points) + num_outer_xs_points + k
                p2 = i*(num_outer_xs_points + num_inner_xs_points) + num_outer_xs_points + (k+1)
                p3 = (i+1)*(num_outer_xs_points + num_inner_xs_points) + num_outer_xs_points + (k+1)
                p4 = (i+1)*(num_outer_xs_points + num_inner_xs_points) + num_outer_xs_points + k

                if (k == num_inner_xs_points - 1):
                    p2 = i*(num_outer_xs_points + num_inner_xs_points) + num_outer_xs_points
                    p3 = (i+1)*(num_outer_xs_points + num_inner_xs_points) + num_outer_xs_points
                
                side_faces[i*(num_outer_xs_points + num_inner_xs_points) + num_outer_xs_points + k, :] = [4, p1, p2, p3, p4]
        
        print(side_faces)
        # create end faces from cross-section polydata
        bottom_faces = node_xsections[0].faces
        top_faces = np.copy(node_xsections[-1].faces)
        for i in range(0, len(top_faces), 4):
            top_faces[i+1] += (num_outer_xs_points + num_inner_xs_points)*(self.n-1)
            top_faces[i+2] += (num_outer_xs_points + num_inner_xs_points)*(self.n-1)
            top_faces[i+3] += (num_outer_xs_points + num_inner_xs_points)*(self.n-1)

        
        # concatenate faces into one array - PyVista reads them in as a flat array
        all_faces = np.hstack([side_faces.flatten(), bottom_faces, top_faces])
        # all_faces = np.hstack([top_faces])
        
        # create and plot the mesh with PyVista
        surf = pv.PolyData(vertices, all_faces)
        surf.triangulate(inplace=True)
        return surf

    def _getCrossSectionFaces(self, outer_xsection_points, inner_xsection_points):
        # faces will be defined as triplets of integer indices
        # indices > number of outer cross section points correspond to points in the inner cross section
        _, num_outer_xs_points = outer_xsection_points.shape
        _, num_inner_xs_points = inner_xsection_points.shape

        # iterate through outer cross-section points and find closest point on the inner cross section to it
        outer_closest_points = []
        for i in range(num_outer_xs_points):
            cur_p = outer_xsection_points[:,i]
            sub = np.subtract(inner_xsection_points, cur_p[:, np.newaxis])
            sub_norm = np.linalg.norm(sub, axis=0)
            index = np.argmin(sub_norm)
            outer_closest_points.append(index)
        
        # now make the faces
        # iterate through the outer points
        faces = []
        for i in range(num_outer_xs_points):
            next_i = (i+1) % num_outer_xs_points

            cur_cp = outer_closest_points[i]
            next_cp = outer_closest_points[next_i]

            if cur_cp == next_cp:
                faces.append(np.array([3, i, num_outer_xs_points + cur_cp, next_i]))
            
            else:
                # check for the case that we wrap around the inner cross section indices
                if next_cp < cur_cp:
                    for k in range(cur_cp, num_inner_xs_points):
                        next_k = (k+1) % num_inner_xs_points

                        faces.append(np.array([3, i, num_outer_xs_points + k, num_outer_xs_points + next_k]))
                        
                    for k in range(0,next_cp):
                        faces.append(np.array([3, i, num_outer_xs_points + k, num_outer_xs_points + next_k]))
                
                else:
                    # normal case: next_cp > cur_cp
                    for k in range(cur_cp, next_cp):
                        next_k = (k+1) % num_inner_xs_points
                        faces.append(np.array([3, i, num_outer_xs_points + k, num_outer_xs_points + next_k]))
                
                faces.append(np.array([3, i, num_outer_xs_points + next_cp, next_i]))
        
        return np.ravel(faces)


    # returns the cross sections of the rod at each node along the rod as a pv.PolyData object
    def nodeCrossSectionPolyData(self, scale_factor=1.0):

        # get transforms from base to each node
        node_transforms = self.nodeTransforms()

        # create vertices for undeformed cross-section
        outer_xsection_points = self.cross_section.xs_outer.meshPoints()
        inner_xsection_points = self.cross_section.xs_inner.meshPoints()
        for p in outer_xsection_points:
            p *= scale_factor
        for p in inner_xsection_points:
            p *= scale_factor
        
        inner_xsection_points[0,:] += self.cross_section.xa
        inner_xsection_points[1,:] += self.cross_section.ya

        _, num_outer_xs_points = outer_xsection_points.shape
        _, num_inner_xs_points = inner_xsection_points.shape

        faces = self._getCrossSectionFaces(outer_xsection_points, inner_xsection_points)

        # list for all the poly data
        nodes_poly_data = []

        for i,transform in enumerate(node_transforms):
            R = transform[0:3, 0:3]
            p = transform[0:3, 3]

            C = self.nodeDistortionMatrix(i)
            # deform and transform the base circle into the current frame
            cur_outer_xsection_points = np.matmul(R, np.matmul(C, outer_xsection_points)) + p[:, np.newaxis] # use np.newaxis to explicitly create a column vector
            cur_inner_xsection_points = np.matmul(R, np.matmul(C, inner_xsection_points)) + p[:, np.newaxis] # use np.newaxis to explicitly create a column vector
            
            cur_xsection_points = np.hstack((cur_outer_xsection_points, cur_inner_xsection_points))
            # outer_face = list(range(num_outer_xs_points))
            # outer_face.insert(0,num_outer_xs_points)

            # inner_face = list(range(num_inner_xs_points))
            # inner_face.insert(0,num_inner_xs_points)

            # outer_poly_data = pv.PolyData(cur_outer_xsection_points.transpose(), outer_face)
            # inner_poly_data = pv.PolyData(cur_inner_xsection_points.transpose(), inner_face)
            # combined_poly_data = outer_poly_data.triangulate()# + inner_poly_data.triangulate()
            poly_data = pv.PolyData(cur_xsection_points.transpose(), faces)
            nodes_poly_data.append(poly_data)
        return nodes_poly_data

    def nodeCrossSectionPolyData2D(self):
        outer_xsection_points = self.cross_section.xs_outer.meshPoints()
        inner_xsection_points = self.cross_section.xs_inner.meshPoints()

        _, num_outer_xs_points = outer_xsection_points.shape
        _, num_inner_xs_points = inner_xsection_points.shape

        inner_xsection_points[0,:] += self.cross_section.xa
        inner_xsection_points[1,:] += self.cross_section.ya

        faces = self._getCrossSectionFaces(outer_xsection_points, inner_xsection_points)

        nodes_poly_data_2d = []
        for i in range(self.n):

            C = self.nodeDistortionMatrix(i)
            # deform and transform the base circle into the current frame
            cur_outer_xsection_points = np.matmul(C, outer_xsection_points)
            cur_inner_xsection_points = np.matmul(C, inner_xsection_points)
            cur_xsection_points = np.hstack((cur_outer_xsection_points, cur_inner_xsection_points))
            # outer_face = list(range(num_outer_xs_points))
            # outer_face.insert(0,num_outer_xs_points)

            # inner_face = list(range(num_inner_xs_points))
            # inner_face.insert(0,num_inner_xs_points)

            # outer_poly_data = pv.PolyData(cur_outer_xsection_points.transpose(), outer_face)
            # inner_poly_data = pv.PolyData(cur_inner_xsection_points.transpose(), inner_face)
            # combined_poly_data = outer_poly_data.triangulate()# + inner_poly_data.triangulate()
            poly_data = pv.PolyData(cur_xsection_points.transpose(), np.array(faces))
            nodes_poly_data_2d.append(poly_data)

        return nodes_poly_data_2d

    # returns the distortion matrix C for node i
    def nodeDistortionMatrix(self, i):
        # extract variables
        a = self.Z[0:self.n]
        b = self.Z[self.n:2*self.n]
        c = self.Z[2*self.n:3*self.n]

        C = np.array([[a[i], c[i], 0], [c[i], b[i], 0], [0, 0, 1]])

        return C
    
    def tipPosition(self, ab_coords=[0,0]):
        end_transform = self.nodeTransforms()[-1]
        p = end_transform[0:3, 3]

        C = self.nodeDistortionMatrix(-1)
        deformed_ab_coords = np.matmul(C[0:2,0:2], np.array(ab_coords))
        x_dir = end_transform[0:3,0]
        y_dir = end_transform[0:3,1]
        return p # + x_dir * deformed_ab_coords[0] * self.cross_section.rx  + y_dir * deformed_ab_coords[1] * self.cross_section.ry



###############################################################################
# Tendon-actuated cosserat rod
###############################################################################
class TendonActuatedCosseratRod:
    def __init__(self, n, L, cross_section, tendons, E, nu):
        self.n = n
        self.L = L
        self.cross_section = cross_section
        self.tendons = tendons

        # set material properties
        self.E = E
        self.nu = nu

        M = E * (1-nu) / ( (1+nu) * (1-2*nu))
        lam = E * nu / ( (1+nu) * (1-2*nu))
        self.M = M
        self.G = E / (2 * (1+nu))       # shear modulus
        self.K = np.matrix([[ M, lam, lam, 0, 0, 0 ],   # stiffness matrix for Cosserat rod
                            [ lam, M, lam, 0, 0, 0 ],
                            [ lam, lam, M, 0, 0, 0 ],
                            [ 0, 0, 0, self.G, 0, 0 ],
                            [ 0, 0, 0, 0, self.G, 0 ],
                            [ 0, 0, 0, 0, 0, self.G ]])
        
        # initialize Z
        # define initial variables for optimization problem (variables in the undeformed state)
        #_ cross-sectional paramaters a, b, c are defined at each of the n nodes along the length
        a_0 = np.ones(n)      # a = 1 when cross-section is undeformed
        b_0 = np.ones(n)      # b = 1 when cross-section is undeformed
        c_0 = np.zeros(n)     # c = 0 when cross-section is undeformed

        # link parameters are defined at the midpoint between the nodes, i.e. there are n-1 links
        v1_0 = np.zeros(n-1)  # v1 = 0 when no loads on rod
        v2_0 = np.zeros(n-1)  # v2 = 0 when no loads on rod
        v3_0 = np.ones(n-1)   # v3 = 1 when no loads on rod
        u1_0 = np.zeros(n-1)  # u1 = 0 when no loads on rod
        u2_0 = np.zeros(n-1)  # u2 = 0 when no loads on rod
        u3_0 = np.zeros(n-1)  # u3 = 0 when no loads on rod

        self.Z = np.hstack([a_0, b_0, c_0, v1_0, v2_0, v3_0, u1_0, u2_0, u3_0])

    def solveOptimizationProblem(self, applied_tip_forces):
        # put some bounds/constraints on Z
        # bounds of (None,None) indicate that a variable has no bounds in the optimization
        Z_bounds = [(None,None)]*self.Z.size # array of None's the size of Z

        # constrain cross-section at base of rod (s=0) to be undeformed
        Z_bounds[0] = (1,1)     # a at s=0 is 1 (undeformed)
        Z_bounds[self.n] = (1,1)     # b at s=0 is 1 (undeformed)
        Z_bounds[2*self.n] = (0,0)   # c at s=0 is 0 (undeformed)

        # constrain cross-section at end of rod (s=L) to be undeformed
        Z_bounds[self.n-1] = (1,1)   # a at s=L is 1 (undeformed)
        Z_bounds[2*self.n-1] = (1,1) # b at s=L is 1 (undeformed)
        Z_bounds[3*self.n-1] = (0,0) # c at s=L is 0 (undeformed)

        # constrain the rest of a, b, and c to be reasonable values (shouldn't be needed)
        # for i in range(1,self.n-1):
        #     Z_bounds[i] = (0, 10)
        #     Z_bounds[self.n + i] = (0, 10)
        #     Z_bounds[2*self.n + i] = (-10, 10)

        # constrain v1, v2, v3 to be reasonable values (shouldn't be needed)
        # for i in range(0, self.n-1):
        #     Z_bounds[3*self.n + i] = (0,0)
        #     Z_bounds[3*self.n + self.n-1 + i] = (0,0)
        #     Z_bounds[3*self.n + 2*(self.n-1) + i] = (0.5,2)

        # constrain u1, u2, u3 to be reasonable values (shouldn't be needed)
        # for i in range(0, self.n-1):
        #     Z_bounds[3*self.n + 3*(self.n-1) + i] = (0,0)
        #     Z_bounds[3*self.n + 4*(self.n-1) + i] = (0,0)
        #     Z_bounds[3*self.n + 5*(self.n-1) + i] = (0,0)

        res = opt.minimize(self._totalEnergy, self.Z, args=(applied_tip_forces), method='slsqp', bounds=Z_bounds, options={'maxiter': 10000, 'disp':True})
        self.Z = res.x

        a =  self.Z[0:self.n]
        b =  self.Z[self.n:2*self.n]
        c =  self.Z[2*self.n:3*self.n]
        v1 = self.Z[3*self.n:4*self.n-1]
        v2 = self.Z[4*self.n-1:5*self.n-2]
        v3 = self.Z[5*self.n-2:6*self.n-3]
        u1 = self.Z[6*self.n-3:7*self.n-4]
        u2 = self.Z[7*self.n-4:8*self.n-5]
        u3 = self.Z[8*self.n-5:9*self.n-6]

        print(f"a: {a}")
        print(f"b: {b}")
        print(f"c: {c}")

    def _totalEnergy(self, Z, applied_tip_forces):
        h = self.L / (self.n - 1)

        energy = 0
        g = np.eye(4) # current transform along the rod g(s)

        # extract variables
        a = Z[0:self.n]
        b = Z[self.n:2*self.n]
        c = Z[2*self.n:3*self.n]
        v1 = Z[3*self.n:4*self.n-1]
        v2 = Z[4*self.n-1:5*self.n-2]
        v3 = Z[5*self.n-2:6*self.n-3]
        u1 = Z[6*self.n-3:7*self.n-4]
        u2 = Z[7*self.n-4:8*self.n-5]
        u3 = Z[8*self.n-5:9*self.n-6]

        tendon_lengths = np.zeros((len(self.tendons),))

        # iterate through the links (i.e. midpoints between the nodes)
        for i in range(0,self.n-1):
            # get cross-sectional parameters at midpoint between nodes
            a_mid = (a[i] + a[i+1])/2
            b_mid = (b[i] + b[i+1])/2
            c_mid = (c[i] + c[i+1])/2

            # get approximate derivates of a, b, and c
            a_prime = (a[i+1] - a[i])/h
            b_prime = (b[i+1] - b[i])/h
            c_prime = (c[i+1] - c[i])/h

            # compute energy for this segment
            k11 = self.E*(self.cross_section.Ix*b_mid**2 + self.cross_section.Iy*c_mid**2)
            k22 = self.E*(self.cross_section.Iy*a_mid**2 + self.cross_section.Ix*c_mid**2)
            k12 = -self.E*(self.cross_section.Iy*a_mid*c_mid + self.cross_section.Ix*b_mid*c_mid)
            k33 = self.G*self.cross_section.Iy*(a_mid**2 + c_mid**2) + self.G*self.cross_section.Ix*(b_mid**2 + c_mid**2)
            K_mat = np.matrix([ [k11, k12, 0], [k12, k22, 0], [0, 0, k33] ])
            u_vec = np.array([u1[i], u2[i], u3[i]])
            strain_vec = np.array([a_mid - 1, b_mid -1, v3[i] - 1])
            M_mat = self.K[0:3, 0:3]
            energy += 0.5 * h * (self.G*self.cross_section.A0*v1[i]**2 + self.G*self.cross_section.A0*v2[i]**2 + 4*self.G*self.cross_section.A0*c_mid**2 +
                                 np.matmul(np.matmul(u_vec, K_mat), u_vec) +
                                 self.cross_section.A0 * np.matmul(np.matmul(strain_vec, M_mat), strain_vec) +
                                 self.G*self.cross_section.Iy*a_prime**2 + self.G*self.cross_section.Ix*b_prime**2 + self.G*(self.cross_section.Ix + self.cross_section.Iy)*c_prime**2 +
                                 2*self.G*u3[i]*( (a_mid*c_prime - a_prime*c_mid)*self.cross_section.Iy - (b_mid*c_prime - b_prime*c_mid)*self.cross_section.Ix)
                                 )
            # compute energy stored in this segment
            # energy += 0.5 * h * (e0_energy + ex_energy + ey_energy)
            
            for ti,tendon in enumerate(self.tendons):
                rho = np.array([[tendon.location[0]], [tendon.location[1]], [0]])
                u = np.array([u1[i], u2[i], u3[i]])
                C_mat = np.matrix([[a_mid, c_mid, 0], [c_mid, b_mid, 0], [0, 0, 1]])
                C_mat_prime = np.matrix([[a_prime, c_prime, 0], [c_prime, b_prime, 0], [0, 0, 1]])
                
                l_prime = np.array([v1[i], v2[i], v3[i]]) + np.matmul(utils.Skew3(u), np.matmul(C_mat, rho)).T + np.matmul(C_mat_prime, rho).T
                
                if tendon.location[2] < i*h:
                    continue
                elif tendon.location[2] >= i*h and tendon.location[2] <= (i+1)*h:
                    tendon_lengths[ti] += np.linalg.norm(l_prime) * (tendon.location[2] - i*h)
                else:
                    tendon_lengths[ti] += np.linalg.norm(l_prime) * h

            # compute the transform from the base to node i
            g = np.matmul(g, utils.MatExp_se3([h*u1[i], h*u2[i], h*u3[i], h*v1[i], h*v2[i], h*v3[i]]))

        # print tendon lengths
        print(tendon_lengths)

        for length,tendon in zip(tendon_lengths, self.tendons):
            energy += length * tendon.tension

        x_dir = g[0:3, 0]  # tip transform x-direction (a coord is in this direction)
        y_dir = g[0:3, 1]  # tip transform y-direction (b coord is in this direction)
        tip_pos = g[0:3, 3]
        
        for tip_force in applied_tip_forces:
            p = tip_pos + x_dir * tip_force.location[0] + y_dir * tip_force.location[1] # position at end of rod where force is applied

            if tip_force.dead_load:
                force = tip_force.force
            else:
                R = g[0:3, 0:3]
                force = np.matmul(R, tip_force.force)   # rotate force from body coords into global frame
            # print(np.dot(force,p))
            # subtract the applied force from the total internal energy
            energy -= np.dot(force, p)
            
        return energy.item() # use .item() to turn the 1x1 matrix into a scalar

    # returns a list of 4x4 transformation matrices (starting with node 0, i.e. the base node) from the base to each node
    def nodeTransforms(self):
        h = self.L / (self.n-1)

        # extract variables
        v1 = self.Z[3*self.n:4*self.n-1]
        v2 = self.Z[4*self.n-1:5*self.n-2]
        v3 = self.Z[5*self.n-2:6*self.n-3]
        u1 = self.Z[6*self.n-3:7*self.n-4]
        u2 = self.Z[7*self.n-4:8*self.n-5]
        u3 = self.Z[8*self.n-5:9*self.n-6]

        # create vertices for undeformed cross-section
        xsection_points = self.cross_section.meshPoints()
        _, num_xs_points = xsection_points.shape

        # create matrix for all vertices in the mesh
        vertices = np.zeros((3, num_xs_points*self.n))

        transforms = []

        g = np.eye(4)
        transforms.append(g)
        for i in range(self.n-1):
            # extract rotation and position from transformation
            R = g[0:3, 0:3]
            p = g[0:3, 3]

            # compute the transform from the base to node i+1
            g = np.matmul(g, utils.MatExp_se3([h*u1[i], h*u2[i], h*u3[i], h*v1[i], h*v2[i], h*v3[i]]))

            transforms.append(g)
        
        return transforms

    def nodeTransform(self, s):
        h = self.L / (self.n-1)

        # extract variables
        v1 = self.Z[3*self.n:4*self.n-1]
        v2 = self.Z[4*self.n-1:5*self.n-2]
        v3 = self.Z[5*self.n-2:6*self.n-3]
        u1 = self.Z[6*self.n-3:7*self.n-4]
        u2 = self.Z[7*self.n-4:8*self.n-5]
        u3 = self.Z[8*self.n-5:9*self.n-6]

        g = np.eye(4)
        for i in range(self.n-1):
            # extract rotation and position from transformation
            R = g[0:3, 0:3]
            p = g[0:3, 3]

            if s < h*(i+1):
                new_h = h*(i+1) - s
                g = np.matmul(g, utils.MatExp_se3([new_h*u1[i], new_h*u2[i], new_h*u3[i], new_h*v1[i], new_h*v2[i], new_h*v3[i]]))
                break
            else:
                # compute the transform from the base to node i+1
                g = np.matmul(g, utils.MatExp_se3([h*u1[i], h*u2[i], h*u3[i], h*v1[i], h*v2[i], h*v3[i]]))
        
        return g

    def asMesh(self):
        h = self.L / (self.n-1)

        # extract variables
        a = self.Z[0:self.n]
        b = self.Z[self.n:2*self.n]
        c = self.Z[2*self.n:3*self.n]

        node_xsections = self.nodeCrossSectionPolyData()
        num_xs_points = node_xsections[0].points.shape[0] # number of points in each cross section

        # create matrix for all vertices in the mesh
        vertices = np.zeros((num_xs_points*self.n, 3))

        for i,xsection in enumerate(node_xsections):
            # add the points to the overall matrix
            vertices[num_xs_points*(i):num_xs_points*(i+1),:] = xsection.points

        # create 'side' faces for mesh between each cross section
        faces = np.zeros((num_xs_points*(self.n-1), 5), dtype=int)
        for i in range(self.n-1):
            for k in range(num_xs_points):
                p1 = i*num_xs_points + k
                p2 = i*num_xs_points + (k+1)
                p3 = (i+1)*num_xs_points + (k+1)
                p4 = (i+1)*num_xs_points + k
                if (k == num_xs_points - 1):
                    p2 = i*num_xs_points
                    p3 = (i+1)*num_xs_points
                
                faces[i*num_xs_points + k, :] = [4, p1, p2, p3, p4]
        
        # create end faces
        # s=0 end face
        bottom_face = list(range(num_xs_points))
        bottom_face.insert(0, len(bottom_face))

        # s=L end face
        top_face = list(range(num_xs_points*(self.n-1), num_xs_points*self.n))
        top_face.insert(0, len(top_face))
        
        # concatenate faces into one array - PyVista reads them in as a flat array
        all_faces = np.hstack([faces.flatten(), bottom_face, top_face])
        
        # create and plot the mesh with PyVista
        surf = pv.PolyData(vertices, all_faces)
        surf.triangulate(inplace=True)
        return surf

    # returns the cross sections of the rod at each node along the rod as a pv.PolyData object
    def nodeCrossSectionPolyData(self, scale_factor=1.0):

        # get transforms from base to each node
        node_transforms = self.nodeTransforms()

        # create vertices for undeformed cross-section
        xsection_points = self.cross_section.meshPoints()
        for p in xsection_points:
            p *= scale_factor

        _, num_xs_points = xsection_points.shape

        # list for all the poly data
        nodes_poly_data = []

        for i,transform in enumerate(node_transforms):
            R = transform[0:3, 0:3]
            p = transform[0:3, 3]

            C = self.nodeDistortionMatrix(i)
            # deform and transform the base circle into the current frame
            cur_xsection_points = np.matmul(R, np.matmul(C, xsection_points)) + p[:, np.newaxis] # use np.newaxis to explicitly create a column vector
            face = list(range(num_xs_points))
            face.insert(0,num_xs_points)

            poly_data = pv.PolyData(cur_xsection_points.transpose(), face)
            nodes_poly_data.append(poly_data)

        return nodes_poly_data

    def nodeCrossSectionPolyData2D(self):
        xsection_points = self.cross_section.meshPoints()
        _, num_xs_points = xsection_points.shape

        nodes_poly_data_2d = []
        for i in range(self.n):
            C = self.nodeDistortionMatrix(i)
            deformed_xsection_points = np.matmul(C, xsection_points)
            face = list(range(num_xs_points))
            face.insert(0, num_xs_points)

            poly_data = pv.PolyData(deformed_xsection_points.transpose(), face)
            nodes_poly_data_2d.append(poly_data)

        return nodes_poly_data_2d

    # returns the distortion matrix C for node i
    def nodeDistortionMatrix(self, i):
        # extract variables
        a = self.Z[0:self.n]
        b = self.Z[self.n:2*self.n]
        c = self.Z[2*self.n:3*self.n]

        C = np.array([[a[i], c[i], 0], [c[i], b[i], 0], [0, 0, 1]])

        return C
    
    def tipPosition(self, ab_coords=[0,0]):
        end_transform = self.nodeTransforms()[-1]
        p = end_transform[0:3, 3]

        C = self.nodeDistortionMatrix(-1)
        deformed_ab_coords = np.matmul(C[0:2,0:2], np.array(ab_coords))
        x_dir = end_transform[0:3,0]
        y_dir = end_transform[0:3,1]
        return p + x_dir * deformed_ab_coords[0] * self.cross_section.rx  + y_dir * deformed_ab_coords[1] * self.cross_section.ry
    


import cosserat
import mesh
import utils

import pyvista as pv
import copy
import numpy as np

ROD_FILENAME = "cxx/output/peristaltic.txt"

MODEL_COLOR = [255, 130, 0]

def loadRobotFromFilename(filename):
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
    state = np.array(data_arr[7:-3]).astype(float)

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
    
    rod = cosserat.CosseratRod(N, length, cross_section, E, nu)

    deformed_rod = copy.deepcopy(rod)
    deformed_rod.Z = state

    return (rod, deformed_rod)

def plotModel(deformed_rod):
    plotter = pv.Plotter()
    plotter.add_text("Model Results")
    # plotter.camera.position = [0, -5*ROD_WIDTH_X*len(deformed_rods), ROD_LENGTH]

    # get mesh from Cosserat rod class
    mesh = deformed_rod.asMesh()

    
    plotter.add_mesh(mesh, color=MODEL_COLOR, opacity=0.7, specular=1.0, smooth_shading=True, split_sharp_edges=True, show_edges=False)
    # plot cross sections
    deformed_xsections = deformed_rod.nodeCrossSectionPolyData()
    for xsection in deformed_xsections:


        plotter.add_mesh(xsection, color=MODEL_COLOR, opacity=1.0, show_edges=True, edge_color='k')
       

    plotter.add_floor()
    plotter.show()


def main():
    
    undeformed_rod, deformed_rod = loadRobotFromFilename(ROD_FILENAME)
    
    plotModel(deformed_rod)

if __name__ == "__main__":
    main()
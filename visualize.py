
import cosserat
import mesh
import utils

import pyvista as pv

ROD_FILENAME = "cxx/output/peristaltic.txt"

MODEL_COLOR = [255, 130, 0]

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
    
    undeformed_rod, deformed_rod = utils.loadRodFromFile(ROD_FILENAME)
    
    plotModel(deformed_rod)

if __name__ == "__main__":
    main()

import cosserat
import mesh
import utils
import vtk

import os
import time

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

ROD_FILENAME = "cxx/output/peristaltic.txt"
# SIM_FOLDER_PATH = "cxx/output/sim"
# SIM_FOLDER_PATH = "cxx/output/N=13_3circle_sim"
SIM_FOLDER_PATH = "cxx/output/N=109_pipe_sim"

MODEL_COLOR = [255, 130, 0]

def represents_int(s):
    try: 
        int(s)
    except ValueError:
        return False
    else:
        return True

def readSimFromFolder(folder_path):
    filenames = sorted(os.listdir(folder_path))
    robot_filename = os.path.join(folder_path, filenames[0]) # should be robot.txt
    with open(robot_filename, 'r') as file:
        data = file.read()
        data_arr = data.split("\n")
    
    N = int(data_arr[0])
    length = float(data_arr[1])
    E = float(data_arr[2])
    nu = float(data_arr[3])
    cs_type = data_arr[4]
    cs_rx = float(data_arr[5])
    cs_ry = float(data_arr[6])

    if cs_type == "Ellipse":
        cross_section = cosserat.AnalyticalEllipseCrossSection(cs_rx, cs_ry)
    elif cs_type == "Rect":
        cross_section = cosserat.AnalyticalRectCrossSection(cs_rx*2, cs_ry*2)

    rod = cosserat.CosseratRod(N, length, cross_section, E, nu)

    actuator_pressures = []
    positions = []
    orientations = []
    states = []
    for step_filename in filenames[1:]:
        with open(os.path.join(folder_path, step_filename)) as file:

            # extract actuator pressures
            pressures = [int(s) for s in file.readline().split(" ") if represents_int(s)]
            actuator_pressures.append(pressures)

            # extract rod state
            data = file.read()
            data_arr = data.split("\n")

            expected_states = 9*N-6
            state = np.array(data_arr[:expected_states]).astype(float)
            position = np.array(data_arr[expected_states:expected_states+3]).astype(float)
            if len(data_arr) > expected_states+3:
                orientation = np.array(data_arr[expected_states+3:expected_states+6]).astype(float)
            else:
                orientation = np.array([0,0,0])
            states.append(state)
            positions.append(position)
            if np.linalg.norm(orientation) < 1e-12:
                orientations.append(np.eye(3))
            else:
                orientations.append(utils.MatExp_so3(orientation/np.linalg.norm(orientation), np.linalg.norm(orientation)))

    return rod, actuator_pressures, positions, orientations, states
       
    

def main():
    

    (rod, actuator_pressures, positions, orientations, states) = readSimFromFolder(SIM_FOLDER_PATH)

    step_start = 0
    rod_mesh = rod.asMesh(rod.n//2, positions[step_start], orientations[step_start])


    ##########################################
    # Set up plotter
    ###########################################
    pl = pv.Plotter()
    pl.camera.tight = False
    pl.enable_anti_aliasing()
    # pl.camera.enable_parallel_projection()
    # pl.camera_position = 'xy'
    pl.camera_position = [1,0,10]
    pl.camera.focal_point = [1.01, 0, 0]
    pl.camera.up = [0,1,0]
    # pl.camera.zoom(0.3)
    # pl.camera.position = [0, 0, 1]
    # pl.camera.clipping_range = (0.01, 1000.01)

    ######################
    # Pipe sim stuff
    ######################
    pl.camera_position = [-5, 0, 0.5]
    pl.camera.focal_point = [0, 0, 0.49]

    cylinder = pv.Cylinder(radius=0.2, height=5, resolution=50, direction=(0,0,1))
    cylinder = cylinder.triangulate()
    inner_cylinder = pv.Cylinder(radius=0.12, height=5, resolution=50, direction=(0,0,1))
    inner_cylinder = inner_cylinder.triangulate()
    pipe = cylinder - inner_cylinder
    half_pipe = pipe.clip('x', value=0, invert=False)
    pl.add_mesh(half_pipe, color=[150,150,150])

    ############################################
    # Create initial meshes (that will be updated each time step)
    ############################################

    # the rod
    pv_mesh = pv.PolyData(rod_mesh.points, faces=rod_mesh.faces)
    actor = pl.add_mesh(pv_mesh, color=MODEL_COLOR, opacity=0.7, specular=1.0)

    # one for each cross section
    xsections = rod.nodeCrossSectionPolyData(rod.n//2, positions[step_start], orientations[step_start])
    cross_section_meshes = []
    for i,xsection in enumerate(xsections):
        pv_cs_mesh = pv.PolyData(xsection.points, faces=xsection.faces)
        cross_section_meshes.append(pv_cs_mesh)
        pl.add_mesh(cross_section_meshes[i], color=MODEL_COLOR, opacity=1.0, show_edges=True, edge_color='k')
    
    # pl.add_floor()

    # axes = pl.add_axes_at_origin()

    ##############################
    # Path following stuff
    ##############################
    r1 = 4  
    c1 = np.array([0,r1])
    # circle1 = pv.Circle(r1)
    # for pt in circle1.points:
    #     pt[0] = pt[0] + c1[0]
    #     pt[1] = pt[1] + c1[1]
    # # pl.add_mesh(circle1)

    r2 = 3.0
    c2 = c1 + r1*np.array([np.sqrt(2)/2, -np.sqrt(2)/2]) + r2*np.array([np.sqrt(2)/2, -np.sqrt(2)/2])
    # circle2 = pv.Circle(r2)
    # for pt in circle2.points:
    #     pt[0] = pt[0] + c2[0]
    #     pt[1] = pt[1] + c2[1]
    # # pl.add_mesh(circle2, color=[255,0,0])#, style='wireframe')

    r3 = 2.5
    c3 = c2 + (r2+r3)*np.array([1, 0])
    # circle3 = pv.Circle(r3)
    # for pt in circle3.points:
    #     pt[0] = pt[0] + c3[0]
    #     pt[1] = pt[1] + c3[1]
    # pl.add_mesh(circle3, color=[0,255,0])
    
    # def create_alternating_arc(center, radius, start_angle, end_angle, 
    #                       segment_length=0.1, visible_ratio=0.6):
    #     """Create arc with alternating visible/invisible segments"""
    #     num_segments = int(radius * np.abs(end_angle - start_angle) / segment_length)
    #     angles = np.linspace(start_angle, end_angle, num_segments + 1)
    #     lines = []
        
    #     for i in range(num_segments):
    #         if i % 2 == 0:  # Only create every other segment
    #             segment_angles = np.linspace(angles[i], angles[i] + 
    #                                     (angles[i+1] - angles[i]) * visible_ratio, 10)
    #             x = center[0] + radius * np.cos(segment_angles)
    #             y = center[1] + radius * np.sin(segment_angles)
    #             z = np.full_like(x, 0)
                
    #             points = np.column_stack([x, y, z])
    #             lines.append(pv.lines_from_points(points))
        
    #     return lines

    # # Create alternating arc
    # seg_l = 0.35
    # vr = 0.8
    # alt_segments1 = create_alternating_arc(center=c1, radius=r1, 
    #                                     start_angle=-np.pi/2, end_angle=-np.pi/4,
    #                                     segment_length=seg_l, visible_ratio=vr)
    # alt_segments2 = create_alternating_arc(center=c2, radius=r2,
    #                                       start_angle=3*np.pi/4, end_angle=0,
    #                                       segment_length=seg_l, visible_ratio=vr)
    # alt_segments3 = create_alternating_arc(center=c3, radius=r3,
    #                                        start_angle=-np.pi, end_angle=-np.pi/2,
    #                                        segment_length=seg_l, visible_ratio=vr)

    # for segment in alt_segments1:
    #     pl.add_mesh(segment, color='gray', line_width=6)
    # for segment in alt_segments2:
    #     pl.add_mesh(segment, color='gray', line_width=6)
    # for segment in alt_segments3:
    #     pl.add_mesh(segment, color='gray', line_width=6)

    # speed_text_actor = pl.add_text('1x', position='upper_right', color='black', font_size=28, font='times')


    # plotting still frames from video
    # pl.camera_position = [4,0,25]
    # pl.camera.focal_point = [3.99, 0, 0]
    # pl.camera.up = [0,1,0]
    # # update the rod state and get its new mesh
    # steps_to_plot = [20, 600, 1200, 1802]
    # for step in steps_to_plot:
    #     rod.Z = states[step]
    #     new_rod_mesh = rod.asMesh(rod.n//2, positions[step], orientations[step])

    #     new_pv_mesh = pv.PolyData(new_rod_mesh.points, faces=new_rod_mesh.faces)
    #     # pv_mesh.shallow_copy(new_pv_mesh)
    #     pl.add_mesh(new_pv_mesh, color=MODEL_COLOR, opacity=0.7, specular=1.0)

    #     # update the cross section meshes
    #     new_xsections = rod.nodeCrossSectionPolyData(rod.n//2, positions[step], orientations[step])
    #     for i,xsection in enumerate(new_xsections):
    #         new_pv_cs_mesh = pv.PolyData(xsection.points, faces=xsection.faces)
    #         # cross_section_meshes[i].shallow_copy(new_pv_cs_mesh)
    #         pl.add_mesh(new_pv_cs_mesh, color=MODEL_COLOR, opacity=1.0, show_edges=True, edge_color='k')


    # error plot
    # def distanceToCircle(p, c, r):
    #     diff = p - c
    #     return np.abs( np.linalg.norm(diff) - r)
    # errors = []
    # def getError(p):
    #     d1 = distanceToCircle(p, c1, r1)
    #     d2 = distanceToCircle(p, c2, r2)
    #     d3 = distanceToCircle(p, c3, r3)
    #     return min([d1,d2,d3])
    
    # for pos in positions:
    #     errors.append(getError(np.array([pos[0], pos[1]])))
    
    # fig = plt.figure(figsize=(8,1))
    # plt.tight_layout()
    # plt.plot(range(0,1978-step_start), np.array(errors[step_start:1978]))
    # plt.xlabel('Step Num')
    # plt.ylabel('Abs Error')
    # plt.show()
    # return


    # start plotting
    pl.show(auto_close=False, interactive_update=True, window_size=[1920,1080])
    # pl.show()

    time.sleep(10)

    #########################################
    # Animate through the sim states
    #########################################

    # loop through the simulation states and update the meshes
    wait_time = 1/20 # s
    for step in range(step_start,len(states)):
        print(step)
        # update the rod state and get its new mesh
        rod.Z = states[step]
        new_rod_mesh = rod.asMesh(rod.n//2, positions[step], orientations[step])

        new_pv_mesh = pv.PolyData(new_rod_mesh.points, faces=new_rod_mesh.faces)
        pv_mesh.shallow_copy(new_pv_mesh)

        # update the cross section meshes
        new_xsections = rod.nodeCrossSectionPolyData(rod.n//2, positions[step], orientations[step])
        for i,xsection in enumerate(new_xsections):
            new_pv_cs_mesh = pv.PolyData(xsection.points, faces=xsection.faces)
            cross_section_meshes[i].shallow_copy(new_pv_cs_mesh)

        # # path drawing for path following sim
        # if wait_time == 1/100:
        #     if step%10==0:
        #         segment = pv.Line(positions[step-10], positions[step])
        #         pl.add_mesh(segment, color='red', line_width=2)
        # else:
        #     segment = pv.Line(positions[step-1], positions[step])
        #     pl.add_mesh(segment, color='red', line_width=2)

        # # time control for path following sim 
        # if step == 100:
        #     pl.remove_actor(speed_text_actor)
        #     speed_text_actor = pl.add_text('5x', position='upper_right', color='black', font_size=28, font='times')
        #     wait_time = 1/100
        # if step == 1900:
        #     pl.remove_actor(speed_text_actor)
        #     speed_text_actor = pl.add_text('1x', position='upper_right', color='black', font_size=28, font='times')
        #     wait_time = 1/20
        # if step > 1978:
        #     break

        

        # print(step)

        # camera_x = min( 1+max(step-40,0)/200, 5)
        # camera_z = min(5+max(step-40,0)/60, 18)
        # pl.camera.position = [camera_x,0,camera_z]
        # pl.camera.focal_point = [camera_x-0.01, 0, 0]
        # pl.camera.up = [0,1,0]
        # pl.reset_camera_clipping_range()

        pl.render()
        time.sleep(wait_time)

    time.sleep(10)
    pl.close()

if __name__ == "__main__":
    main()
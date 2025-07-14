# CosseratRod
Energy minimization for static solution of Cosserat rods with cross-sectional deformation.

## Extracting FEA data from COMSOL
1. Run the FEA simulation
2. Under Results, Right click 'Export', then click 'Data'
3. Under 'Dataset', Select the solution
4. Under 'Expressions', add 3 expressions: x+u, y+v, z+w (u, v, w are the x,y,z displacements)
5. Under 'Output', change 'Geometry Level' to 'Surface' (i.e. only print data for surface nodes)
6. Choose a filename and click 'Export' at the top

The resulting .txt file will have 6 columns: Original X,Y,Z and Deformed X,Y,Z. However, these nodes may be in a different order than the nodes loaded from the .stl file. To fix this, use `utils.getDeformedMeshFromComsolData()`, which will loop through each vertex in the .stl and find the corresponding vertex in the .txt output file.

## Extracting FEA data from Inventor Nastran
0. Download FNO Reader (link https://forums.autodesk.com/t5/inventor-nastran-forum/read-binary-results-file-fno-with-a-program/m-p/9020216)
1. In Inventor Nastran, right-click "Results" and click "Show in folder" which will take you to the location of the output files from the analysis
2. Generating .stl File
   - In FNO Reader, select "NAS to CAD" option
   - Input the .nas filename with the same name as the analysis output file (should be in the same folder as your .fno output file from step 1)
   - Hit 'Next' until you get to enter the output filename
   - Enter output filename and click "Create the Output"
3. Generating the undeformed nodes .csv file
   - In FNO Reader, select "NAS to Text" option
   - Input the .nas filename from step 2b, click "Next"
   - Scroll down until you see the row for "GRID", and check the checkbox next to it
   - In the drop-down menu at the top, change "All rows" to "Checked rows only", click "Next"
   - Enter output filename and click "Create the Output"
4. Generating the node displacements .csv file
   - In FNO Reader, select "FNO to Table" option
   - Enter the .fno filename from step 1, click "Next"
   - Under "Number to Output", select all rows (scroll to bottom and Shift+Click to highlight all at once) and click the "<" button
   - Select "[2] T1 TRANSLATION", "[3] T2 TRANSLATION", and "[4] T3 TRANSLATION" (using Ctrl+Click) and clikc the ">" button. There should just be these 3 outputs on the right side
   - Click "Next", enter the output filename, and click "Create the Output"

Now you have all the information you need to construct the deformed surface mesh. The undeformed .stl file gives us the surface vertices and connectivity information, and we can associate nodes from the undeformed nodes .csv file with the .stl vertices to find correspondance between the .stl vertices and the Nastran internal node numbering. Once we have that, we can get the appropriate nodal displacements from the node displacements .csv file that we can apply to the undeformed .stl mesh. This process is implemented in `utils.getDeformedMeshFromNastranData()`.

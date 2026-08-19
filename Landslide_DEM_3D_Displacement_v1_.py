import os
import sys
import math
import datetime
import numpy as np
import cv2
from osgeo import gdal

# Allow GDAL to throw Python exceptions
gdal.UseExceptions()
# =============================================================================
# Information
# Purpose: generate 3D landlside displacment geotiffs using two dates of DEMs
#
# Input DEMs must be same CRS and resolution, 
# but do not have to be the same size/region.
#
# Installation & Environment Setup
# 
# 1. Open your Anaconda Prompt.
# 2. Create the environment and install GDAL and NumPy via conda-forge:
#    conda create --name dem_tracking -c conda-forge python=3.9 gdal numpy -y
# 3. Activate the environment:
#    conda activate dem_tracking
# 4. Install the headless version of OpenCV using pip (this prevents .dll GUI conflicts with GDAL):
#    python -m pip install opencv-python-headless
# 5. Edit this script USER INPUTS and save
# 6. Save the script and run it from your activated Anaconda Prompt:
#    C: (Change to the drive with script)
#    cd path/to/script/folder (change directory with script)
#    python Landslide_DEM_3D_Displacment_v1_.py (run the script)

# =============================================================================
# =============================================================================
# 1. USER INPUTS
# =============================================================================
DEM1_PATH = r"C:\Landslide3D\DEMs\2020_Landslide_DEM_50cm.tif"
DEM2_PATH = r"C:\Landslide3D\DEMs\2026_Landslide_DEM_50cm.tif"
OUTPUT_DIR = r"C:\Landslide3D\Outputs\2020-2026"

# Dates for the two DEMs to calculate rates (Format: YYYY-MM-DD)
DATE_DEM1 = "2020-08-01"
DATE_DEM2 = "2026-08-01"

# Maximum expected horizontal displacement in meters (used to auto-size search window)
MAX_DISP_M = 10.0 

# Grid spacing for output in meters (e.g., calculate a vector every 10 meters)
GRID_SPACING_M = 4.0

# Azimuth filter (0 to 360 degrees). Flow direction must fall within this wedge.
# Example: If flow is generally South-West, you might use 180 to 270.
# To disable filtering and allow all directions, use 0 and 360.
MIN_AZIMUTH = 0.0
MAX_AZIMUTH = 360.0

# Rate unit choice: "m/day" or "m/year"
RATE_UNIT = "m/year"

# Chip sizes in pixels. Can be an integer or "auto".
# "auto" for half_chip defaults to 15 pixels. 
# "auto" for search_chip sizes the window based on MAX_DISP_M.
HALF_CHIP = 'auto' 
SEARCH_CHIP = 'auto'

# Minimum correlation coefficient to keep the result (0.0 to 1.0)
MIN_CORRELATION = 0.4

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================
def calculate_azimuth(dx_m, dy_m):
    """Calculate geographic azimuth (0-360) where North is 0, East is 90."""
    azimuth = math.degrees(math.atan2(dx_m, dy_m))
    return (azimuth + 360) % 360

def is_within_azimuth(az, min_az, max_az):
    """Check if azimuth falls within the target wedge, handling 360 degree wrap."""
    if min_az <= max_az:
        return min_az <= az <= max_az
    else: # Crosses North (e.g., 350 to 10)
        return az >= min_az or az <= max_az

# =============================================================================
# 3. MAIN SCRIPT
# =============================================================================
def main():
    print("Starting 3D DEM Pixel Tracking...")
    
    # 1. Error Checking & Setup
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    if not os.path.exists(DEM1_PATH) or not os.path.exists(DEM2_PATH):
        print("ERROR: One or both input DEM paths do not exist.")
        sys.exit(1)
        
    ds1_raw = gdal.Open(DEM1_PATH)
    ds2_raw = gdal.Open(DEM2_PATH)
    
    # 2. Calculate bounding boxes (Extents)
    def get_extent(ds):
        gt = ds.GetGeoTransform()
        x_min = gt[0]
        x_max = gt[0] + ds.RasterXSize * gt[1]
        y_max = gt[3] # Usually the top
        y_min = gt[3] + ds.RasterYSize * gt[5] # gt[5] is negative
        return [x_min, min(y_min, y_max), x_max, max(y_min, y_max)]
        
    ext1 = get_extent(ds1_raw)
    ext2 = get_extent(ds2_raw)
    
    # Find the overlapping intersection
    intersect_x_min = max(ext1[0], ext2[0])
    intersect_x_max = min(ext1[2], ext2[2])
    intersect_y_min = max(ext1[1], ext2[1])
    intersect_y_max = min(ext1[3], ext2[3])
    
    if intersect_x_min >= intersect_x_max or intersect_y_min >= intersect_y_max:
        print("ERROR: The two DEMs do not overlap spatially.")
        sys.exit(1)
        
    out_bounds = [intersect_x_min, intersect_y_min, intersect_x_max, intersect_y_max]
    
    pixel_width = abs(ds1_raw.GetGeoTransform()[1])
    pixel_height = abs(ds1_raw.GetGeoTransform()[5])
    
    # 3. Warp both DEMs in-memory to the exact same overlapping grid
    print("Aligning and cropping DEMs to their overlapping extent in memory...")
    warp_options = gdal.WarpOptions(
        format='MEM',
        outputBounds=out_bounds,
        xRes=pixel_width,
        yRes=pixel_height,
        dstSRS=ds1_raw.GetProjection(),
        resampleAlg=gdal.GRA_Bilinear
    )
    
    ds1 = gdal.Warp('', ds1_raw, options=warp_options)
    ds2 = gdal.Warp('', ds2_raw, options=warp_options)
    gt1 = ds1.GetGeoTransform()
    
    # 4. Create derivatives directly in memory
    print("Generating slope derivatives in memory...")
    slope_ds1 = gdal.DEMProcessing('', ds1, 'slope', format='MEM', computeEdges=True)
    slope_ds2 = gdal.DEMProcessing('', ds2, 'slope', format='MEM', computeEdges=True)
    
    # Load the aligned data into arrays
    print("Loading aligned data into memory arrays...")
    z1 = ds1.GetRasterBand(1).ReadAsArray()
    z2 = ds2.GetRasterBand(1).ReadAsArray()
    slp1 = slope_ds1.GetRasterBand(1).ReadAsArray().astype(np.float32)
    slp2 = slope_ds2.GetRasterBand(1).ReadAsArray().astype(np.float32)
    
    rows, cols = z1.shape
    
    # Calculate temporal difference
    d1 = datetime.datetime.strptime(DATE_DEM1, "%Y-%m-%d")
    d2 = datetime.datetime.strptime(DATE_DEM2, "%Y-%m-%d")
    days_diff = (d2 - d1).days
    
    if days_diff <= 0:
        print("ERROR: DATE_DEM2 must be after DATE_DEM1.")
        sys.exit(1)
        
    if RATE_UNIT == "m/day":
        time_scalar = 1.0 / days_diff
        suffix = "mperday"
    elif RATE_UNIT == "m/year":
        time_scalar = 365.25 / days_diff
        suffix = "mperyr"
    else:
        print("ERROR: Invalid RATE_UNIT.")
        sys.exit(1)
        
    # Set up chip sizes
    global HALF_CHIP, SEARCH_CHIP
    
    # Clean up HALF_CHIP and force integer
    if str(HALF_CHIP).strip().lower() == 'auto':
        HALF_CHIP = 15 # Default 15x15 pixel half-chip
    else:
        HALF_CHIP = int(HALF_CHIP)
        
    # Clean up SEARCH_CHIP and force integer
    if str(SEARCH_CHIP).strip().lower() == 'auto':
        max_px_disp = int(math.ceil(MAX_DISP_M / pixel_width))
        SEARCH_CHIP = HALF_CHIP + max_px_disp
    else:
        SEARCH_CHIP = int(SEARCH_CHIP)
        
    # Grid steps in pixels
    grid_step_x = max(1, int(GRID_SPACING_M / pixel_width))
    grid_step_y = max(1, int(GRID_SPACING_M / pixel_height))
    
    # Calculate proper output dimensions for the new coarser grid
    x_indices = list(range(SEARCH_CHIP, cols - SEARCH_CHIP, grid_step_x))
    y_indices = list(range(SEARCH_CHIP, rows - SEARCH_CHIP, grid_step_y))
    
    out_cols = len(x_indices)
    out_rows = len(y_indices)
    
    # Prepare output arrays mapped to the new coarser dimensions
    out_dx = np.full((out_rows, out_cols), np.nan, dtype=np.float32)
    out_dy = np.full((out_rows, out_cols), np.nan, dtype=np.float32)
    out_dz = np.full((out_rows, out_cols), np.nan, dtype=np.float32)
    out_3d_raw = np.full((out_rows, out_cols), np.nan, dtype=np.float32) # For Band 1
    out_3d = np.full((out_rows, out_cols), np.nan, dtype=np.float32)     # For Band 2
    out_az = np.full((out_rows, out_cols), np.nan, dtype=np.float32)
    out_cc = np.full((out_rows, out_cols), np.nan, dtype=np.float32)
    
    # Generate new GeoTransform for the coarser outputs
    out_gt = list(gt1)
    out_gt[1] = gt1[1] * grid_step_x  
    out_gt[5] = gt1[5] * grid_step_y  
    out_gt[0] = gt1[0] + (x_indices[0] * gt1[1])  
    out_gt[3] = gt1[3] + (y_indices[0] * gt1[5])  
    
    # 5. Cross-Correlation Processing
    print(f"Processing tracking grid (Spacing: {GRID_SPACING_M}m) ...")
    valid_matches = 0
    filtered_by_limit = 0
    
    # Loop over the grid tracking both input high-res coordinates and output low-res coordinates
    for out_y, y in enumerate(y_indices):
        for out_x, x in enumerate(x_indices):
            
            # Extract template from DEM1 derivative
            template = slp1[y - HALF_CHIP : y + HALF_CHIP + 1, 
                            x - HALF_CHIP : x + HALF_CHIP + 1]
            
            # Extract search window from DEM2 derivative
            search_window = slp2[y - SEARCH_CHIP : y + SEARCH_CHIP + 1, 
                                 x - SEARCH_CHIP : x + SEARCH_CHIP + 1]
                                 
            # Skip nodata blocks
            if np.isnan(template).any() or np.isnan(search_window).any():
                continue
                
            # Perform Template Matching
            res = cv2.matchTemplate(search_window, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            cc = max_val
            if cc < MIN_CORRELATION:
                continue
                
            # Calculate pixel offsets
            center_offset = SEARCH_CHIP - HALF_CHIP
            px_dx = max_loc[0] - center_offset
            px_dy = max_loc[1] - center_offset 
            
            # Convert pixel offset to metric displacement
            dx_m = px_dx * pixel_width
            dy_m = px_dy * -pixel_height # Map Y increases upwards geographically
            
            azimuth = calculate_azimuth(dx_m, dy_m)
            
            if not is_within_azimuth(azimuth, MIN_AZIMUTH, MAX_AZIMUTH):
                continue
                
            # Calculate Elevation (Z) Component
            z_start = z1[y, x]
            tracked_y = y + px_dy
            tracked_x = x + px_dx
            
            # Prevent out of bounds on the second DEM due to max displacement tracking
            if tracked_y < 0 or tracked_y >= rows or tracked_x < 0 or tracked_x >= cols:
                continue
                
            z_end = z2[tracked_y, tracked_x]
            dz_m = z_end - z_start
            
            # Calculate Total 3D Displacement (raw meters)
            total_3d_m = math.sqrt(dx_m**2 + dy_m**2 + dz_m**2)
            
            # Filter against MAX_DISP_M in any direction
            # This eliminates false "edge" matches that snap to the search window boundary
            if abs(dx_m) > MAX_DISP_M or abs(dy_m) > MAX_DISP_M or abs(dz_m) > MAX_DISP_M or total_3d_m > MAX_DISP_M:
                filtered_by_limit += 1
                continue
            
            # Apply Temporal Scalar and write to the Coarser Output Arrays
            out_dx[out_y, out_x] = dx_m * time_scalar
            out_dy[out_y, out_x] = dy_m * time_scalar
            out_dz[out_y, out_x] = dz_m * time_scalar
            out_3d_raw[out_y, out_x] = total_3d_m           # Band 1: Raw Meters
            out_3d[out_y, out_x] = total_3d_m * time_scalar # Band 2: Scaled Rate
            out_az[out_y, out_x] = azimuth
            out_cc[out_y, out_x] = cc
            
            valid_matches += 1

    print(f"Tracking complete.")
    print(f"Valid vectors found: {valid_matches}")
    print(f"Vectors filtered exceeding max displacement ({MAX_DISP_M}m): {filtered_by_limit}")
    
    # 6. Write Outputs
    print("Writing output GeoTIFFs...")
    driver = gdal.GetDriverByName('GTiff')
    
    def write_geotiff(filename, array_data, nodata=np.nan):
        filepath = os.path.join(OUTPUT_DIR, filename)
        out_ds = driver.Create(filepath, out_cols, out_rows, 1, gdal.GDT_Float32)
        out_ds.SetGeoTransform(out_gt)
        out_ds.SetProjection(ds1.GetProjection())
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(array_data)
        out_band.SetNoDataValue(nodata)
        out_band.FlushCache()
        out_ds = None
        
    write_geotiff(f"displacement_3D_{suffix}.tif", out_3d)
    write_geotiff(f"displacement_X_{suffix}.tif", out_dx)
    write_geotiff(f"displacement_Y_{suffix}.tif", out_dy)
    write_geotiff(f"displacement_Z_{suffix}.tif", out_dz)
    write_geotiff(f"flow_azimuth.tif", out_az)
    write_geotiff(f"correlation_coeff.tif", out_cc)

    # Write Multi-band GeoTIFF
    mb_filename = f"displacement_multiband_{suffix}.tif"
    mb_filepath = os.path.join(OUTPUT_DIR, mb_filename)
    mb_ds = driver.Create(mb_filepath, out_cols, out_rows, 4, gdal.GDT_Float32)
    mb_ds.SetGeoTransform(out_gt)
    mb_ds.SetProjection(ds1.GetProjection())
    
    # Band 1
    b1 = mb_ds.GetRasterBand(1)
    b1.SetDescription("Magnitude")
    b1.WriteArray(out_3d_raw) # Writing raw meters
    b1.SetNoDataValue(np.nan)
    
    # Band 2
    b2 = mb_ds.GetRasterBand(2)
    b2.SetDescription(suffix) # Evaluates to "mperyr" or "mperday"
    b2.WriteArray(out_3d) # Writing scaled rate
    b2.SetNoDataValue(np.nan)
    
    # Band 3
    b3 = mb_ds.GetRasterBand(3)
    b3.SetDescription("Azimuth")
    b3.WriteArray(out_az)
    b3.SetNoDataValue(np.nan)
    
    # Band 4
    b4 = mb_ds.GetRasterBand(4)
    b4.SetDescription("Corr")
    b4.WriteArray(out_cc)
    b4.SetNoDataValue(np.nan)
    
    mb_ds.FlushCache()
    mb_ds = None

    # 7. Write Log / Metadata File
    log_path = os.path.join(OUTPUT_DIR, "run_metadata.txt")
    print(f"Writing run log to {log_path}...")
    with open(log_path, 'w') as f:
        f.write("3D DEM PIXEL TRACKING RUN REPORT\n")
        f.write("================================\n")
        f.write(f"Run Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("INPUTS:\n")
        f.write(f"DEM 1: {DEM1_PATH} (Date: {DATE_DEM1})\n")
        f.write(f"DEM 2: {DEM2_PATH} (Date: {DATE_DEM2})\n")
        f.write(f"Time Interval: {days_diff} days\n\n")
        f.write("PARAMETERS:\n")
        f.write(f"Maximum Expected Displacement: {MAX_DISP_M} meters (Used for search window AND hard cutoff filter)\n")
        f.write(f"Grid Spacing: {GRID_SPACING_M} meters\n")
        f.write(f"Azimuth Filter: {MIN_AZIMUTH} to {MAX_AZIMUTH} degrees\n")
        f.write(f"Rate Unit: {RATE_UNIT}\n")
        f.write(f"Correlation Threshold: {MIN_CORRELATION}\n\n")
        f.write("COMPUTED CHIP SIZES (Pixels):\n")
        f.write(f"Half-Chip (Template Radius): {HALF_CHIP}\n")
        f.write(f"Search-Chip (Search Radius): {SEARCH_CHIP}\n\n")
        f.write("OUTPUT DEFINITIONS:\n")
        f.write(f"Output Grid Size: {GRID_SPACING_M}m x {GRID_SPACING_M}m\n")
        f.write(f"1. displacement_3D_{suffix}.tif: Total 3D vector length (sqrt(X^2 + Y^2 + Z^2)) in {RATE_UNIT}.\n")
        f.write(f"2. displacement_X_{suffix}.tif: East-West displacement component in {RATE_UNIT}.\n")
        f.write(f"3. displacement_Y_{suffix}.tif: North-South displacement component in {RATE_UNIT}.\n")
        f.write(f"4. displacement_Z_{suffix}.tif: Vertical (elevation change) component in {RATE_UNIT}. Computed as DEM2(end) - DEM1(start).\n")
        f.write(f"5. flow_azimuth.tif: Direction of horizontal movement in degrees (0=North, 90=East).\n")
        f.write(f"6. correlation_coeff.tif: OpenCV TM_CCOEFF_NORMED match score (0 to 1).\n")
        f.write(f"7. displacement_multiband_{suffix}.tif: 4-Band composite layer.\n")
        f.write(f"     - Band 1: 'Magnitude' (Raw total 3D displacement distance in meters).\n")
        f.write(f"     - Band 2: '{suffix}' (Scaled 3D displacement rate in {RATE_UNIT}).\n")
        f.write(f"     - Band 3: 'Azimuth' (Direction of horizontal movement in degrees).\n")
        f.write(f"     - Band 4: 'Corr' (OpenCV Match score from 0 to 1).\n")
        f.write(f"\nRESULTS:\n")
        f.write(f"Total valid vectors generated: {valid_matches}\n")
        f.write(f"Vectors filtered for exceeding max displacement ({MAX_DISP_M}m): {filtered_by_limit}\n")
        
    # Free memory
    ds1 = None
    ds2 = None
    ds1_raw = None
    ds2_raw = None
    slope_ds1 = None
    slope_ds2 = None
    
    print("Process Finished Successfully!")

if __name__ == "__main__":
    main()

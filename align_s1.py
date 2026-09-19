import rasterio
import shutil

s2_path = "s2.tif"
s1_path = "s1.tif"

with rasterio.open(s2_path) as src2:
    kwargs = src2.meta.copy()
    bounds = src2.bounds
    transform = src2.transform
    crs = src2.crs

with rasterio.open(s1_path) as src1:
    data = src1.read()
    
kwargs.update({
    'count': data.shape[0],
    'dtype': data.dtype,
    'width': data.shape[2],
    'height': data.shape[1],
    'transform': transform,
    'crs': crs
})

with rasterio.open("s1_aligned.tif", "w", **kwargs) as dst:
    dst.write(data)

print("Aligned s1.tif to s2.tif's CRS and Transform. Saved as s1_aligned.tif")

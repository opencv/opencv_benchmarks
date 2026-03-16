# Script to download Scanned Objects by Google Research dataset and Stanford models
# Distributed by CC-BY 4.0 License

# Dataset page:
# https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research

import sys, argparse, subprocess
from pathlib import Path
import requests
import zipfile, tarfile, gzip
import csv, json
import math
import numpy as np

try:
    import cv2 as cv
except ImportError:
    raise ImportError("Failed to import OpenCV, please check that cv2 library is in PYTHONPATH env variable, "
                      "for example use the path {opencv_build_dir}/lib/python3/")

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    raise Exception("Python 3.5 or greater is required. Try running `python3 download_collection.py`")

# global params
verbose = False

class ModelData:
    def __init__(self, name : str, description : str, filesize : int, thumb_url : str, categories ) -> None:
        self.name = name
        self.description = description
        self.filesize = filesize
        self.thumb_url = thumb_url
        self.categories = set(categories)

def print_size(num):
    if num < 1024:
        return str(num) + " B"
    elif num < 1 << 20:
        return "%.3f KiB" % (num / 1024)
    elif num < 1 << 30:
        return "%.3f MiB" % (num / (1 << 20))
    else:
        return "%.3f GiB" % (num / (1 << 30))

collection_name = "Scanned Objects by Google Research"
owner_name = "GoogleResearch"

base_url ='https://fuel.gazebosim.org/'
fuel_version = '1.0'

def download_model(model_name, dir):
    if verbose:
        print()
        print("{}: {}".format(model.name, model.description))
        print("Categories: [", ", ".join(model.categories), "]")
        print("Size:", print_size(model.filesize))

    download_url = base_url + fuel_version + '/{}/models/'.format(owner_name) + model_name + '.zip'

    archive_path = Path(dir) / Path(model_name+'.zip')
    tmp_archive_path   = Path(dir) / Path(model_name+'.zip.tmp')
    mesh_path     = Path(dir) / Path(model_name+'.obj')
    tmp_mesh_path = Path(dir) / Path(model_name+'.obj.tmp')
    mtl_path     = Path(dir) / Path('model.mtl')
    tmp_mtl_path = Path(dir) / Path('model.mtl.tmp')
    texture_path     = Path(dir) / Path('texture.png')
    tmp_texture_path = Path(dir) / Path('texture.png.tmp')

    for tmp in [tmp_archive_path, tmp_mesh_path, tmp_mtl_path, tmp_texture_path]:
        tmp.unlink(missing_ok=True)

    if archive_path.exists():
        if verbose:
            print("Archive exists")
    else:
        print("URL:", download_url)
        attempt = 1
        while True:
            print("download attempt "+str(attempt)+"...", end="")
            try:
                download = requests.get(download_url, stream=True, timeout=5.0)
                break
            except requests.exceptions.Timeout:
                print("Timed out")
                attempt = attempt + 1
        with open(tmp_archive_path, 'wb') as fd:
            for chunk in download.iter_content(chunk_size=1024*1024):
                fd.write(chunk)
                print(".", end="")
        tmp_archive_path.rename(archive_path)
        print("..downloaded")

    with zipfile.ZipFile(archive_path) as z:
        if mesh_path.exists():
            if verbose:
                print("OBJ exists")
        else:
            with open(tmp_mesh_path, 'wb') as f:
                f.write(z.read("meshes/model.obj"))
            tmp_mesh_path.rename(mesh_path)
            print("OBJ unpacked")
        if texture_path.exists():
            if verbose:
                print("Texture exists")
        else:
            with open(tmp_texture_path, 'wb') as f:
                f.write(z.read("materials/textures/texture.png"))
            tmp_texture_path.rename(texture_path)
            print("Texture unpacked")

    if mtl_path.exists():
        if verbose:
            print("Material exists")
    else:
        mtlFile = """
# Copyright 2020 Google LLC.
# 
# This work is licensed under the Creative Commons Attribution 4.0
# International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by/4.0/ or send a letter
# to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
newmtl material_0
# shader_type beckmann
map_Kd texture.png

# Kd: Diffuse reflectivity.
Kd 1.000000 1.000000 1.000000
"""
        with open(tmp_mtl_path, 'xt') as f:
            f.writelines(mtlFile)
        tmp_mtl_path.rename(mtl_path)
        print("Material written")
    return mesh_path, texture_path


def get_thumb(model : ModelData, dir):
    if verbose:
        print(model.name)
    img_url = base_url + fuel_version + model.thumb_url
    img_path = Path(dir) / Path(model.name+'.jpg')
    tmp_path = Path(dir) / Path(model.name+'.jpg.tmp')
    tmp_path.unlink(missing_ok=True)
    if img_path.exists():
        if verbose:
            print("...exists")
    else:
        print("URL:", img_url)
        attempt = 1
        while True:
            print("download attempt "+str(attempt)+"...")
            try:
                download = requests.get(img_url, stream=True, timeout=5.0)
                break
            except requests.exceptions.Timeout:
                print("Timed out")
                attempt = attempt + 1
        with open(tmp_path, 'wb') as fd:
            for chunk in download.iter_content(chunk_size=1024*1024):
                fd.write(chunk)
                print(".", end="")
        tmp_path.rename(img_path)
        print("..downloaded")


def get_content(content_file):
    # Getting model names and URLs
    models_json = []

    # Iterate over the pages
    page = 0
    while True:
        page = page + 1
        next_url = '/models?page={}&per_page=100&q=collections:{}'.format(page,collection_name)
        page_url = base_url + fuel_version + next_url

        print("Gettting page %d..." % page)
        r = requests.get(page_url)

        if not r or not r.text:
            break

        # Convert to JSON
        models_page = json.loads(r.text)

        if not models_page:
            break

        models_json.extend(models_page)

        print(len(models_json), " models")

    json_object = json.dumps(models_json, indent=4)
    with open(content_file, "w") as outfile:
        outfile.write(json_object)

    return models_json


# let's use different chunk sizes to get rid of timeouts
stanford_models = [
["http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz", 1, "bunny/reconstruction/bun_zipper.ply"],
["http://graphics.stanford.edu/pub/3Dscanrep/happy/happy_recon.tar.gz", 1024, "happy_recon/happy_vrip.ply"],
["http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz", 1024, "dragon_recon/dragon_vrip.ply"],
["http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz", 64, ""],
["http://graphics.stanford.edu/data/3Dscanrep/lucy.tar.gz", 1024, "lucy.ply"],
["http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_dragon.ply.gz", 1024, ""],
["http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_manuscript.ply.gz", 1024, ""],
["http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_statuette.ply.gz", 1024, ""],
]

def get_stanford_model(url : str, name : str, ext: str, dir : str, chunk_size : int, internal_path : str):
    archive_path     = Path(dir) / Path(name+'.'+ext)
    tmp_archive_path = Path(dir) / Path(name+'.'+ext+'.tmp')
    model_path = Path(dir) / Path(name+'.ply')
    tmp_model_path = Path(dir) / Path(name+'.ply.tmp')

    for tmp in [tmp_archive_path, tmp_model_path]:
        tmp.unlink(missing_ok=True)

    if archive_path.exists():
        if verbose:
            print("Archive exists")
    else:
        print("URL:", url)
        attempt = 1
        while True:
            print("download attempt "+str(attempt)+"...", end="")
            try:
                download = requests.get(url, stream=True, timeout=5.0)
                break
            except requests.exceptions.Timeout:
                print("Timed out")
                attempt = attempt + 1
        with open(tmp_archive_path, 'wb') as fd:
            for chunk in download.iter_content(chunk_size=chunk_size*1024):
                fd.write(chunk)
                print(".", end="")
        tmp_archive_path.rename(archive_path)
        print("..downloaded")

    if model_path.exists():
        if verbose:
            print("Model exists")
    else:
        # to reduce memory footprint for big models
        max_size = 1024*1024*16
        print("Extracting..", end="")
        with open(tmp_model_path, 'xb') as of:
            if ext=="tar.gz":
                tar_obj = tarfile.open(archive_path, 'r', encoding='utf-8', errors='surrogateescape')
                try:
                    reader = tar_obj.extractfile(internal_path)
                    while buf := reader.read(max_size):
                        of.write(buf)
                        print(".", end="")
                except Exception as err:
                    print(err)
                tar_obj.close()
            elif ext=="ply.gz":
                with gzip.open(archive_path) as gz:
                    while buf := gz.read(max_size):
                        of.write(buf)
                        print(".", end="")
        tmp_model_path.rename(model_path)
        print("done")
    return model_path, ""


def lookAtMatrixCal(position, lookat, upVector):
    tmp = position - lookat
    norm = np.linalg.norm(tmp)
    w = tmp / norm
    tmp = np.cross(upVector, w)
    norm = np.linalg.norm(tmp)
    u = tmp / norm
    v = np.cross(w, u)
    res = np.array([
        [u[0], u[1], u[2],   0],
        [v[0], v[1], v[2],   0],
        [w[0], w[1], w[2],   0],
        [0,    0,    0,   1.0]
    ], dtype=np.float32)
    translate = np.array([
        [1.0,   0,   0, -position[0]],
        [0, 1.0,   0, -position[1]],
        [0,   0, 1.0, -position[2]],
        [0,   0,   0,          1.0]
    ], dtype=np.float32)
    res = np.matmul(res, translate)
    return res


# walk over all triangles, collect per-vertex normals
def generateNormals(points, indices):
    success = False
    maxTri = 8
    while not success and maxTri < 64:
        preNormals = np.zeros((points.shape[0], maxTri, 3), dtype=np.float32)
        nNormals = np.zeros((points.shape[0],), dtype=np.int)

        pArr = [ np.take(points, indices[:, j], axis=0) for j in range(3) ]

        crossArr = np.cross(pArr[1] - pArr[0], pArr[2] - pArr[0])
        # normalizing, fixing div by zero
        crossNorm = np.linalg.norm(crossArr, axis=1)
        crossNorm[abs(crossNorm) < 1e-6 ] = 1
        crossNorm3 = np.vstack([crossNorm, crossNorm, crossNorm]).T
        crossArr = crossArr * (1.0 / crossNorm3)

        needMoreTri = False
        for i in range(crossArr.shape[0]):
            tri = indices[i, :]
            cross = crossArr[i, :]

            for j in range(3):
                idx = tri[j]
                n = nNormals[idx]
                if n == maxTri:
                    print(maxTri, " triangles per vertex is not enough, adding 8 more")
                    needMoreTri = True
                    break
                nNormals[idx] = n+1
                preNormals[idx, n, :] = cross
            if needMoreTri:
                break

        if needMoreTri:
            maxTri += 8
        else:
            normals = np.sum(preNormals, axis=1)
            # normalizing, fixing div by zero
            norm = np.linalg.norm(normals, axis=1)
            norm[abs(norm) < 1e-6 ] = 1
            norm3 = np.vstack([norm, norm, norm]).T
            normals = normals * (1.0 / norm3)
            success = True

    if success:
        print ("normals generated")
        return normals
    else:
        raise RuntimeError("too much triangle per vertex")


def writeCsv(path, stat_data):
    with open(path, 'w', newline='') as csvfile:
        fieldnames = [ "model", "normL2Rgb", "normInfRgb", "nzDepthDiff",
                    "normL2Depth", "normInfDepth"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for name, sd in stat_data.items():
            # dict merge operator | is not supported until 3.9
            sdname = sd
            sdname["model"] = name
            writer.writerow(sdname)

# ==================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="bench3d", description="3D algorithm benchmark for OpenCV",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog="The tool works with the latest OpenCV 5.x, "
                                     "a proper path to OpenCV library should be provided in PYTHONPATH env. variable, "
                                     "usually it is {opencv_build_dir}/lib/python3/\n"
                                     "The script first downloads dataset contents to {work_dir}/contents.json "
                                     "then downloads and unpacks the models and their thumbnails. "
                                     "If this process is interrupted, you can start the script again and "
                                     "it will continue from the same point.\n"
                                     "Next, the benchmark for triangleRasterize() starts: each model is "
                                     "rendered by both OpenCV and OpenGL reference tool, results are compared.\n"
                                     "The statistics is saved to {work_dir}/stat.json and {work_dir}/stat.csv "
                                     "after each model processed.\n"
                                     "NOTE: statistics is erased at each script start\n"
                                     "NOTE #2: stanford_* models are quite big, may require several minutes "
                                     "and up to 12 Gb RAM per each")
    parser.add_argument("--work_dir", required=True, help="a directory to store downloaded dataset and generated files")
    parser.add_argument("--verbose", action="store_true",
                        help="indicates whether to display more information to the console or not")
    parser.add_argument("--renderer_path", required=True,
                        help="path to OpenGL tool to render models, usually it is "
                        "{opencv_build_dir}/bin/example_opengl_opengl_testdata_generator")
    parser.add_argument("--debug_pics_dir",
                        help="optional, a directory to store temporary images "
                        "showing in realtime what is being rendered right now")

    args = parser.parse_args()

    if args.verbose:
        verbose = True

    dirname = args.work_dir

    debug_pics_dir = ""
    if args.debug_pics_dir:
        debug_pics_dir = args.debug_pics_dir

    renderer_path = args.renderer_path

    all_models = []

    print("Getting Google Research models")

    content_file = Path(dirname) / Path("content.json")
    if content_file.exists():
        with open(content_file, "r") as openfile:
            models_json = json.load(openfile)
    else:
        Path(dirname).mkdir(parents=True, exist_ok=True)
        models_json = get_content(content_file)

    models = []
    for model in models_json:
        model_name = model['name']
        desc  = model['description']
        fsize = model['filesize']
        thumb_url  = model['thumbnail_url']
        if 'categories' in model:
            categories = model['categories']
        else:
            categories = [ ]
        models.append(ModelData(model_name, desc, fsize, thumb_url, categories))

    print("Getting thumbnail images")
    for model in models:
        get_thumb(model, dirname)

    print("Downloading models from the {}/{} collection.".format(owner_name, collection_name))

    for model in models:
        model_dir = Path(dirname) / Path(model.name)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        model_path, texture_path = download_model(model.name, model_dir)
        all_models.append((model.name, model_path, texture_path))

    print('Done.')

    categories = set()
    for model in models:
        for cat in model.categories:
            categories.add(cat)
    print("Categories:", categories)
    # 'Consumer Goods', 'Bag', 'Car Seat', 'Keyboard', 'Media Cases', 'Toys',
    # 'Action Figures', 'Bottles and Cans and Cups', 'Shoe', 'Legos', 'Hat',
    # 'Mouse', 'Headphones', 'Stuffed Toys', 'Board Games', 'Camera'

    print("\nGetting Stanford models")

    for m in stanford_models:
        url, chunk_size, internal_path = m

        s = url.split("/")[-1].split(".")
        name = "stanford_"+s[0]
        ext = s[1]+"."+s[2]

        if verbose:
            print(name + ":")
        model_dir = Path(dirname) / Path(name)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        model_path, texture_path = get_stanford_model(url, name, ext, model_dir, chunk_size, internal_path)
        all_models.append((name, model_path, texture_path))

    print("\ntriangleRasterize() test")

    width, height = 640, 480
    fovDegrees = 45.0
    fovY = fovDegrees * math.pi / 180.0

    stat_data = {}

    stat_file = Path(dirname) / Path("stat.json")
    csv_file  = Path(dirname) / Path("stat.csv")

    for file in [stat_file, csv_file]:
        file.unlink(missing_ok=True)

    if stat_file.exists():
        with open(stat_file, "r") as openfile:
            stat_data = json.load(openfile)
        writeCsv(csv_file, stat_data)

    for model_name, model_fname, texture_fname in all_models:
        print(model_name)

        colvert_path = Path(model_fname).parent / Path("colvert.ply")
        
        remap_path = Path(debug_pics_dir) / Path("remap.png")
        debug_color_path = Path(debug_pics_dir) / Path("color_raster.png")
        debug_depth_path = Path(debug_pics_dir) / Path("depth_raster.png")

        color_raster_path = Path(model_fname).parent / Path("color_raster.png")
        depth_raster_path = Path(model_fname).parent / Path("depth_raster.png")

        rgb_gl_path   = Path(model_fname).parent / Path("color_gl.png")
        depth_gl_path = Path(model_fname).parent / Path("depth_gl.png")

        color_diff_path = Path(model_fname).parent / Path("color_diff.png")

        depth_diff_path  = Path(model_fname).parent / Path("depth_diff.png")
        mask_gl_path     = Path(model_fname).parent / Path("mask_gl.png")
        mask_raster_path = Path(model_fname).parent / Path("mask_raster.png")

        for file in [ colvert_path, remap_path,
                      color_raster_path, depth_raster_path,
                      rgb_gl_path, depth_gl_path,
                      color_diff_path, depth_diff_path, mask_gl_path, mask_raster_path ]:
            file.unlink(missing_ok=True)

        if verbose:
            print("temp files removed")

        verts, list_indices, normals, colors, texCoords = cv.loadMesh(model_fname)
        if verbose:
            print("loading finished")

        if texture_fname:
            texture = cv.imread(texture_fname) / 255.0

        verts = verts.squeeze()

        # list_indices is a tuple of 1x3 arrays of dtype=int32
        if not list_indices:
            raise ValueError("empty index list")
        for ind in list_indices:
            if ind.shape != (1, 3):
                raise ValueError("wrong index shape")

        indices = np.array(list_indices, dtype=np.int32).squeeze()

        if len(indices.shape) != 2 or (indices.shape[1] != 3):
            raise ValueError("wrong index shape")

        if normals is not None:
            normals = normals.squeeze()

        if colors is not None:
            colors = colors.squeeze()

        minx, miny, minz = np.min(verts[:, 0]), np.min(verts[:, 1]), np.min(verts[:, 2])
        maxx, maxy, maxz = np.max(verts[:, 0]), np.max(verts[:, 1]), np.max(verts[:, 2])

        if verbose:
            print("verts: ", verts.shape)
            print("indices: ", indices.shape)
            print("normals: ", (normals.shape if normals is not None else 0))
            print("colors: ", (colors.shape if colors is not None else 0))
            print("texCoords: ", (texCoords.shape if texCoords is not None else 0))
            print("bounding box: [%f...%f, %f...%f, %f...%f]" % (minx, maxx, miny, maxy, minz, maxz))

        nverts = verts.shape[0]

        # this could be used for slow and dirty texturing
        doRemap = False
        remapCoords = np.ones((nverts, 3), dtype=np.float32)

        if colors is None:
            if texCoords is not None:
                # sample vertex color from texture
                if doRemap:
                    colors = np.ones((nverts, 3), dtype=np.float32)
                    for i in range(nverts):
                        remapCoords[i, :] = [texCoords[i, 0], 1.0-texCoords[i, 1], 0]

                uArr = (       texCoords[:, 0]  * texture.shape[1] - 0.5).astype(np.int)
                vArr = ((1.0 - texCoords[:, 1]) * texture.shape[0] - 0.5).astype(np.int)
                colors = (texture[vArr, uArr, :]).astype(np.float32)

            else:
                if nverts < 1_000_000:
                    # generate normals and set colors to normals
                    normals = generateNormals(verts, indices)
                    colors = abs(normals)
                else:
                    t = np.arange(0, nverts, dtype=np.float32)
                    colors = 0.5 + 0.5 * np.vstack([np.cos(t * (2 * math.pi / nverts)),
                                                    np.sin(t * (3 * math.pi / nverts)),
                                                    np.sin(t * (2 * math.pi / nverts))]).astype(np.float32).T
                    if verbose:
                        print("colors generated")

        # save mesh for OpenGL renderer

        vertsToSave = np.expand_dims(verts, axis=0)
        colorsToSave = np.expand_dims(colors, axis=0)
        colorsToSave = colorsToSave[:, :, ::-1]
        indicesToSave = []
        for i in range(indices.shape[0]):
            ix = indices[i, :]
            indicesToSave.append(ix)

        cv.saveMesh(colvert_path, vertsToSave, indicesToSave, None, colorsToSave)

        if verbose:
            print("mesh saved for OpenGL renderer")

        # rasterize

        ctgY = 1./math.tan(fovY / 2.0)
        ctgX = ctgY / width * height
        zat = maxz + max([abs(maxy) * ctgY,
                          abs(miny) * ctgY,
                          abs(maxx) * ctgX,
                          abs(minx) * ctgX])

        zNear = zat - maxz
        zFar = zat - minz
        position = np.array([0.0, 0.0, zat], dtype=np.float32)
        lookat   = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        upVector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        cameraPose = lookAtMatrixCal(position, lookat, upVector)
        scaleCoeff = 65535.0 / zFar

        depth_buf = np.ones((height, width), dtype=np.float32) * zFar
        color_buf = np.zeros((height, width, 3), dtype=np.float32)

        settings = cv.TriangleRasterizeSettings().setShadingType(cv.RASTERIZE_SHADING_SHADED)
        settings = settings.setCullingMode(cv.RASTERIZE_CULLING_NONE)

        color_buf, depth_buf = cv.triangleRasterize(verts, indices, colors, color_buf, depth_buf,
                                                    cameraPose, fovY, zNear, zFar, settings)

        if verbose:
            print("rasterized")

        if doRemap:
            remap_color_buf, _ = cv.triangleRasterizeColor(verts, indices, remapCoords, color_buf, depth_buf,
                                                           cameraPose, fovY, zNear, zFar, settings)
            mapx = remap_color_buf[:, :, 0] * texture.shape[1] - 0.5
            mapy = remap_color_buf[:, :, 1] * texture.shape[0] - 0.5
            remapped = cv.remap(texture, mapx, mapy, cv.INTER_LINEAR)
            cv.imwrite(remap_path, remapped * 255.0)

        colorRasterize = color_buf
        depthRasterize = (depth_buf * scaleCoeff)

        if debug_pics_dir:
            cv.imwrite(debug_color_path, color_buf * 255.0)
            cv.imwrite(debug_depth_path, depthRasterize.astype(np.ushort))

        cv.imwrite(color_raster_path, color_buf * 255.0)
        cv.imwrite(depth_raster_path, depthRasterize.astype(np.ushort))

        # send mesh to OpenGL rasterizer

        renderer_args = [renderer_path] + [
                "--modelPath="+str(colvert_path),
                "--custom",
                "--fov="+str(fovDegrees),
                "--posx="+str(position[0]),
                "--posy="+str(position[1]),
                "--posz="+str(position[2]),
                "--lookatx="+str(lookat[0]),
                "--lookaty="+str(lookat[1]),
                "--lookatz="+str(lookat[2]),
                "--upx="+str(upVector[0]),
                "--upy="+str(upVector[1]),
                "--upz="+str(upVector[2]),
                "--resx="+str(width),
                "--resy="+str(height),
                "--zNear="+str(zNear),
                "--zFar="+str(zFar),
                "--scaleCoeff="+str(scaleCoeff),
                # white/flat/shaded
                "--shading=shaded",
                # none/cw/ccw
                "--culling=cw",
                "--colorPath="+str(rgb_gl_path),
                "--depthPath="+str(depth_gl_path),
            ]

        if verbose:
            print(renderer_args)
        p = subprocess.run(renderer_args)

        # compare results

        colorGl = cv.imread(rgb_gl_path)
        colorGl = colorGl.astype(np.float32) * (1.0/255.0)
        colorDiff = np.ravel(colorGl - colorRasterize)
        normInfRgb = np.linalg.norm(colorDiff, ord=np.inf)
        normL2Rgb = np.linalg.norm(colorDiff, ord=2) / (width * height)
        print("rgb L2: %f Inf: %f" % (normL2Rgb, normInfRgb))

        cv.imwrite(color_diff_path, (colorDiff.reshape((height, width, 3)) + 1) * 0.5 * 255.0)

        depthGl = cv.imread(depth_gl_path, cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH).astype(np.float32)
        depthGl /= scaleCoeff
        depthRasterize /= scaleCoeff
        depthDiff = depthGl - depthRasterize

        maskGl = abs(depthGl - zFar) > 1e-6
        maskRaster = abs(depthRasterize - zFar) > 1e-6
        maskDiff = maskGl != maskRaster
        nzDepthDiff = np.count_nonzero(maskDiff)
        print("depth nzdiff: %d" % (nzDepthDiff))

        jointMask = maskRaster & maskGl
        nzJointMask = np.count_nonzero(jointMask)
        if nzJointMask:
            maskedDiff = np.ravel(depthDiff[jointMask])
            normInfDepth = np.linalg.norm(maskedDiff, ord=np.inf)
            normL2Depth = np.linalg.norm(maskedDiff, ord=2) / nzJointMask
        else:
            normInfDepth = math.nan
            normL2Depth = math.nan
        print("depth L2: %f Inf: %f" % (normL2Depth, normInfDepth))

        cv.imwrite(depth_diff_path, ((depthDiff) + (1 << 15)).astype(np.ushort))
        cv.imwrite(mask_gl_path, maskGl.astype(np.ubyte) * 255)
        cv.imwrite(mask_raster_path, maskRaster.astype(np.ubyte) * 255)

        stat_data[str(model_name)] = {
            "normL2Rgb"    : str(normL2Rgb),
            "normInfRgb"   : str(normInfRgb),
            "nzDepthDiff"  : str(nzDepthDiff),
            "normL2Depth"  : str(normL2Depth),
            "normInfDepth" : str(normInfDepth)
        }

        stat_json = json.dumps(stat_data, indent=4)
        with open(stat_file, "w") as outfile:
            outfile.write(stat_json)
        writeCsv(csv_file, stat_data)
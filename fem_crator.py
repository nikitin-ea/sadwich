# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 21:21:29 2024

@author: devoi
"""
import sys
import json
import pythoncom
import PyFemap
import numpy as np

from PyFemap import constants
from itertools import pairwise


def load_param(file_path):
    with open(file_path, mode="r+", encoding="utf-8") as file:
        param = json.load(file)
    return param


def get_app_obj():
    try:
        existObj = pythoncom.connect(PyFemap.model.CLSID)
        app = PyFemap.model(existObj)
    except Exception as e:
        print(e)
        sys.exit('Python is not connected to Femap!')
    else:
        rc = app.feAppMessage(constants.FCM_HIGHLIGHT, 'Python is connected to Femap!')
        print('Python is connected to Femap!')
    return app


def get_pt_by_ID(app, pt_ID=None):
    pt = app.fePoint
    if pt_ID is None:
        pt_ID = pt.NextEmptyID()
    if pt_ID == -1:
        pt_ID = pt.NextEmptyID() - 1

    pt.Get(pt_ID)
    return pt, pt_ID


def get_curve_by_ID(app, pt_ID=None):
    pt = app.feCurve
    if pt_ID is None:
        pt_ID = pt.NextEmptyID()
    if pt_ID == -1:
        pt_ID = pt.NextEmptyID() - 1

    pt.Get(pt_ID)
    return pt, pt_ID


def get_surf_by_ID(app, pt_ID=None):
    pt = app.feSurface
    if pt_ID is None:
        pt_ID = pt.NextEmptyID()
    if pt_ID == -1:
        pt_ID = pt.NextEmptyID() - 1

    pt.Get(pt_ID)
    return pt, pt_ID


def convert_list_to_set(app, lst):
    if not isinstance(lst, list):
        lst = [lst, ]
    feset = app.feSet
    ids = [x.ID for x in lst]
    feset.AddArray(len(ids), ids)
    return feset


def convert_set_to_list(app, feset, obj_type):
    feset.First()
    objs = []
    if obj_type == constants.FT_SURFACE:
        while True:
            surf, _ = get_surf_by_ID(app, feset.CurrentID)
            objs.append(surf)
            if not feset.Next():
                break
    return objs


def create_pts(app, param_core, sign):
    pts = []

    for zi, hi in zip(param_core["zi"], param_core["hi"]):
        pt, pt_ID = get_pt_by_ID(app)
        pt.x = zi
        pt.y = sign * hi / 2
        pt.z = 0.0
        pt.Put(pt_ID)
        pts.append(pt)
    return pts


def connect_pts(app, pts):
    lines = []

    for pt_A, pt_B in pairwise(pts):
        app.feLinePointIDs(pt_A.ID, pt_B.ID, True)
        line = app.feCurve
        line_ID = line.NextEmptyID() - 1
        line.Get(line_ID)
        lines.append(line)
    return lines


def get_line_norm(pt_A, length, normal):
    pt_B, pt_B_ID = get_pt_by_ID(app)
    coord = np.array(pt_A.xyz) - length * normal
    pt_B.xyz = tuple(coord)
    pt_B.Put(pt_B_ID)

    app.feLinePointIDs(pt_A.ID, pt_B.ID, True)

    line_norm = app.feCurve
    line_norm_ID = line_norm.NextEmptyID() - 1
    line_norm.Get(line_norm_ID)
    return line_norm


def create_core(app, lines_top, lines_bot):
    surfs = []
    for line1, line2 in zip(lines_top, lines_bot):
        app.feSurfaceRuled(line1.ID, line2.ID)
        surf, _ = get_surf_by_ID(app, -1)
        surfs.append(surf)
    return surfs


def nonmanifold_all(app):
    all_set = app.feSet
    all_set.AddAll(constants.FT_SOLID)
    app.feSurfaceNonManifoldAdd(all_set.ID)

    solid = app.feSolid
    solid_ID = solid.NextEmptyID() - 1
    solid.Get(solid_ID)
    return solid


def create_face(app, lines, param_face, normal):
    lines_set = convert_list_to_set(app, lines)
    pt_A, pt_A_ID = get_pt_by_ID(app, lines[0].vStdPoint[0])
    length = np.interp(pt_A.z, param_face["zi"], param_face["hi"])
    path = get_line_norm(pt_A, length, normal)
    path_set = convert_list_to_set(app, [path, ])

    rc = app.feSurfaceSweep(path_set.ID, lines_set.ID)

    if rc == constants.FE_FAIL:
        app.feAppMessage(3, "Невозможно создать поверхность!")
    else:
        app.feAppMessage(1, "Поверхность создана!")

    solid = app.feSolid
    solid_ID = solid.NextEmptyID() - 1
    solid.Get(solid_ID)

    surfs_on_solid = app.feSet
    surfs_on_solid.AddRule(solid.ID, constants.FGD_SURFACE_ONSOLID)
    return convert_set_to_list(app, surfs_on_solid, constants.FT_SURFACE)


def create_face_mat(app, param_face):
    face_mat = app.feMatl
    face_mat_ID = face_mat.NextEmptyID()
    face_mat.Get(face_mat_ID)
    face_mat_prop = list(face_mat.mmat)
    face_mat_prop[0] = param_face["Ei"][0]
    face_mat_prop[6] = 0.3
    face_mat.mmat = tuple(face_mat_prop)
    face_mat.Put(face_mat_ID)
    return face_mat


def create_core_mat(app, param_core):
    core_mat = app.feMatl
    core_mat_ID = core_mat.NextEmptyID()
    core_mat.Get(core_mat_ID)
    core_mat.type = 1
    core_mat_prop = list(core_mat.mmat)
    core_mat_prop[0] = param_core["Ei"][0]
    core_mat_prop[1] = param_core["Ei"][0]
    core_mat_prop[3] = param_core["Gi"][0]
    core_mat_prop[6] = 0.3
    core_mat.mmat = tuple(core_mat_prop)
    core_mat.Put(core_mat_ID)
    return core_mat


def create_prop(app, mat, param, color):
    prop = app.feProp
    prop_ID = prop.NextEmptyID()
    prop.Get(prop_ID)
    prop.matlID = mat.ID
    prop.type = constants.FET_L_PLANE_STRAIN
    prop_vals = list(prop.pmat)
    prop_vals[0] = param["general"]["b"]
    prop.pmat = tuple(prop_vals)
    prop.color = color
    prop.Put(prop_ID)
    return prop


def get_mesh_size(app, solid, param):
    min_top_size = min(param["top_face"]["hi"])
    min_bot_size = min(param["bot_face"]["hi"])
    min_core_size = min(param["core"]["hi"])
    sizes = sorted([min_top_size, min_bot_size, min_core_size])
    size = sizes[0] / 4 if sizes[0] > sizes[1] / 4 else sizes[1] / 4
    app.feAppMessage(2, f"Размер КЭ назначен равным {size:7.5f} мм.")
    return size


def set_mesh_size(app, mesh_size, solid):
    surfs_set = app.feSet
    surfs_set.AddRule(solid.ID, constants.FGD_SURFACE_ONSOLID)

    replace_all = True
    min_edge = 0
    max_angle = 0.0
    max_on_small = 0
    small_size = 0.0
    vertex_aspect = 0.0
    mapped_refinement = False
    growth_factor = 0.0
    refine_ratio = 0.0
    refine_surf = False
    app.feMeshSizeSurface(surfs_set.ID, replace_all, mesh_size, min_edge,
                          max_angle, max_on_small, small_size, vertex_aspect,
                          mapped_refinement, growth_factor, refine_ratio,
                          refine_surf)


def create_centroid_node(app, node_set, x_shift=0.0):
    x = 0.0
    y = 0.0
    z = 0.0
    node_set.First()
    while True:
        node = app.feNode
        node.Get(node_set.CurrentID)
        x += node.x
        y += node.y
        z += node.z
        if not node_set.Next():
            break
    node_ID = node.NextEmptyID()
    node.Get(node_ID)
    node.x = x / node_set.Count() + x_shift
    node.y = y / node_set.Count()
    node.z = z / node_set.Count()
    node.Put(node_ID)
    return node


def set_left_bc(app, param):
    node_set = app.feSet
    if param["left_end"]["bc"][-1] == "+":
        node_set.AddCoordinate2(constants.FT_NODE, 0, (True, False, False),
                                (4, 0, 0), (0.0, 0.0, 0.0),
                                (1e-3, 1e-5, 1e-5))
    else:
        c = param["core"]["hi"][0] / 2
        node_set.AddCoordinate2(constants.FT_NODE, 0, (True, True, False),
                                (4, 3, 0), (0.0, -c, 0.0),
                                (1e-5, c, 1e-5))

    rbe2 = app.feElem
    rbe2_ID = rbe2.NextEmptyID()
    rbe2.Get(rbe2_ID)
    rbe2.type = constants.FET_L_RIGID
    rbe2.topology = constants.FTO_RIGIDLIST
    rbe2_nodes = list(rbe2.Nodes)
    rbe2_nodes[0] = create_centroid_node(app, node_set, x_shift=-1.0).ID
    rbe2.Nodes = tuple(rbe2_nodes)
    rbe2_dof = list(rbe2.vrelease)
    rbe2_dof[:5] = [1, 1, 1, 1, 1]
    rbe2.vrelease = tuple(rbe2_dof)
    rbe2.PutNodeList(0, node_set.Count(), node_set.GetArray()[2], None, None)
    rbe2.Put(rbe2_ID)


def create_mesh(app, surfs, prop):
    surfs_set = convert_list_to_set(app, surfs)
    app.feMeshSurface2(surfs_set.ID, prop.ID, 4, True, False)
    view = app.feView
    view_ID = view.NextEmptyID()
    view.Get(view_ID)
    view.Initialize()

    vdraw = list(view.vDraw)
    vlabel = list(view.vLabel)
    vcolormode = list(view.vColorMode)

    vdraw[constants.FVI_ELEMENT] = True
    vlabel[constants.FVI_ELEMENT] = False
    vcolormode[constants.FVI_ELEMENT] = 3

    view.vDraw = tuple(vdraw)
    view.vLabel = tuple(vlabel)
    view.vColorMode = tuple(vcolormode)

    view.Visible(True)
    view.FitVisible(True)
    view.Regenerate()
    view.Put(view_ID)


if __name__ == "__main__":
    fname = "kpo3.json"
    param = load_param(fname)
    app = get_app_obj()

    app.feDeleteAll(True, True, True, True)
    pts_top = create_pts(app, param["core"], 1)
    pts_bot = create_pts(app, param["core"], -1)
    lines_top = connect_pts(app, pts_top)
    lines_bot = connect_pts(app, pts_bot)
    surf_face_top = create_face(app, lines_top, param["top_face"],
                                np.array([0.0, -1.0, 0.0]))
    surf_face_bot = create_face(app, lines_bot, param["bot_face"],
                                np.array([0.0, 1.0, 0.0]))
    surf_core = create_core(app, lines_top, lines_bot)
    solid = nonmanifold_all(app)
    mat_face_top = create_face_mat(app, param["top_face"])
    mat_face_bot = create_face_mat(app, param["bot_face"])
    mat_core = create_core_mat(app, param["core"])
    prop_face_top = create_prop(app, mat_face_top, param, constants.FCL_BROWN)
    prop_face_bot = create_prop(app, mat_face_bot, param, constants.FCL_SEPIA)
    prop_core = create_prop(app, mat_core, param, constants.FCL_AVOCADO)
    mesh_size = get_mesh_size(app, solid, param)
    set_mesh_size(app, mesh_size, solid)

    create_mesh(app, surf_face_top, prop_face_top)
    create_mesh(app, surf_face_bot, prop_face_bot)
    create_mesh(app, surf_core, prop_core)

    set_left_bc(app, param)

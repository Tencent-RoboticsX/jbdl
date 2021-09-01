import os
import re
from abc import ABC, abstractmethod
import meshcat.geometry as g
import meshcat.transformations as tf


class Pos:
    # pos data structure
    def __init__(self, xyz=None, rpy=None):
        if xyz is None:
            xyz = [0, 0, 0]
        if rpy is None:
            rpy = [0, 0, 0]
        self.xyz = xyz
        self.rpy = rpy
        self.matrix = self.get_matrix()

    def get_matrix(self):
        matrix = tf.translation_matrix(self.xyz) @ tf.rotation_matrix(self.rpy[2], [0, 0, 1]) @ tf.rotation_matrix(
            self.rpy[1], [0, 1, 0]) @ tf.rotation_matrix(self.rpy[0], [1, 0, 0])
        return matrix


class Visual(ABC):
    def __init__(self, vis, pos=Pos(), name="visual_name"):
        self.name = name
        self.vis = vis
        self.pos = pos

    @abstractmethod
    def render(self):
        pass

    def set_name(self, new_name="visual_new_name"):
        self.name = new_name

    def set_pos(self, new_pos=Pos()):
        self.pos = new_pos


class Mesh(Visual):
    """
    this class is for rendering mesh in xmirror
    support stl,dae,obj
    vis is meshcat visualizer
    name is mesh name
    mesh_path is "package://"+"path from current root"
    you should put description package in the same dir with your scipts
    pos is mesh position
    mesh_type can be stl,dae,obj and can be identify by mesh_path
    """

    def __init__(self,
                 vis,
                 name="",
                 mesh_path="",
                 pos=Pos(),
                 ):
        super(Mesh, self).__init__(vis, pos, name)
        self.mesh_path = mesh_path
        self.mesh_type = "stl"  # can be stl,dae,obj

    def path_join(self, mesh_path=""):
        pattern = "package://"
        path = re.sub(pattern, "", mesh_path)
        current_path = os.path.curdir
        mesh_abspath = os.path.abspath(os.path.join(current_path, path))
        return mesh_abspath

    def mesh_stl_display(self):
        mesh_abspath = self.path_join(self.mesh_path)
        self.vis[self.name].set_object(
            g.StlMeshGeometry.from_file(mesh_abspath))
        self.vis[self.name].set_transform(
            self.pos.matrix
        )

    def mesh_dae_display(self):
        mesh_abspath = self.path_join(self.mesh_path)
        self.vis[self.name].set_object(
            g.DaeMeshGeometry.from_file(mesh_abspath))
        self.vis[self.name].set_transform(
            self.pos.matrix
        )

    def mesh_obj_display(self):
        mesh_abspath = self.path_join(self.mesh_path)
        self.vis[self.name].set_object(
            g.ObjMeshGeometry.from_file(mesh_abspath))
        self.vis[self.name].set_transform(
            self.pos.matrix
        )

    def mesh_type_identify(self):
        stl_pattern = ".stl"
        dae_pattern = ".dae"
        obj_pattern = ".obj"
        if re.search(stl_pattern, self.mesh_path) is not None:
            self.mesh_type = "stl"
        elif re.search(dae_pattern, self.mesh_path) is not None:
            self.mesh_type = "dae"
        elif re.search(obj_pattern, self.mesh_path) is not None:
            self.mesh_type = "obj"

    def render_method(self):
        self.mesh_type_identify()
        if self.mesh_type == "stl":
            self.mesh_stl_display()
        elif self.mesh_type == "dae":
            self.mesh_dae_display()
        elif self.mesh_type == "obj":
            self.mesh_obj_display()

    def render(self):
        # TODO add error exceptions for example:mesh path cannot be none
        self.render_method()


class Box(Visual):
    """
    this class is for rendering box in xmirror
    vis is meshcat visualizer
    name is box name
    size is box 3D size
    material is to give box texture
    pos is box position
    """

    def __init__(self,
                 vis,
                 name="",
                 size=None,
                 material=None,  # TODO
                 pos=Pos(),
                 ):
        super(Box, self).__init__(vis, pos, name)
        self.material = material
        if size is None:
            size = [0.1, 0.1, 0.1]
        self.size = size

    def render(self):
        self.vis[self.name].set_object(
            g.Box(self.size))
        self.vis[self.name].set_transform(
            self.pos.matrix
        )


class Sphere(Visual):
    """
    this class is for rendering sphere in xmirror
    vis is meshcat visualizer
    name is sphere name
    radius is sphere radius
    material is to give sphere texture
    pos is sphere position
    """

    def __init__(self,
                 vis,
                 name="",
                 radius=0.1,
                 material=None,  # TODO
                 pos=Pos(),
                 ):
        super(Sphere, self).__init__(vis, pos, name)
        self.material = material
        self.radius = radius

    def render(self):
        self.vis[self.name].set_object(
            g.Sphere(self.radius))
        self.vis[self.name].set_transform(
            self.pos.matrix
        )


class Cylinder(Visual):
    """
    this class is for rendering cylinder in xmirror the axis of rotational symmetry is aligned with the y-axis.
    vis is meshcat visualizer
    name is cylinder name
    size is cylinder height and radius
    material is to give cylinder texture
    pos is cylinder position
    """

    def __init__(self,
                 vis,
                 name="",
                 size=None,
                 material=None,  # TODO
                 pos=Pos(),
                 ):
        super(Cylinder, self).__init__(vis, pos, name)
        self.material = material
        if size is None:
            size = [0.1, 0.1]
        self.height = size[0]
        self.radius = size[1]

    def render(self):
        self.vis[self.name].set_object(
            g.Cylinder(self.height, self.radius))
        self.vis[self.name].set_transform(
            self.pos.matrix
        )


# class Axis(Visual):
#     # TODO
#     def __init__(self,
#                  vis,
#                  pos,
#                  name=""
#                  ):
#         super(Axis, self).__init__(vis, pos, name)
#         pass
#
#     def render(self):
#         pass
#
#
# def Arrow(Visual):
#     # TODO
#     def __init__(self,
#                  vis,
#                  pos,
#                  name=""
#                  ):
#         super(Arrow, self).__init__(vis, pos, name)
#         pass
#
#
#     def render(self):
#         pass

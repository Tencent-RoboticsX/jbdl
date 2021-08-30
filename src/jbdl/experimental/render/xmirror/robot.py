#!/usr/bin/python3
import numpy as np
import os
import time
import meshcat
import meshcat.transformations as tf
import re
from urdf_parser_py.urdf import URDF
import visual
from visual import Pos
from link_name_tree import LinkNameTree
from world import XML_parser

class RobotModel:
    def __init__(self,
                 vis,
                 name="RobotName",
                 id=0,
                 pos=Pos(),
                 xml_path="",
                 mesh_dir = "None"
                 ):
        # TODO add annotation for this class here
        """

        """
        self.vis = vis
        self.name = name
        self.ID = id
        self.pos = pos
        self.xml_path = xml_path
        self.mesh_dir =mesh_dir
        self.robot_init()

    def robot_init(self):
        if re.search(".urdf", self.xml_path) is not None:
            self.urdf_robot_init()
        elif re.search(".xml", self.xml_path) is not None:
            self.xml_robot_init()

    def urdf_robot_init(self):
        urdf_path = self.xml_path
        self.link_name_tree = LinkNameTree()
        self.pos_tree = {"base": Pos()}
        self.urdf = Urdf_parser(vis=self.vis, link_name_tree=self.link_name_tree, pos_tree=self.pos_tree,
                                urdf_path=urdf_path, robot_name=self.name)
        self.joints = self.urdf.generate_joints()
        self.urdf.generate_link_name_tree()
        self.links = self.urdf.generate_links()

    def xml_robot_init(self):
        pass
        #only if you have just one robot in your xml
        # xml_path = self.xml_path
        # self.link_name_tree = LinkNameTree()
        # self.xml = XML_parser(vis=self.vis,
        #                          xml_path=xml_path,
        #                          mesh_dir=self.mesh_dir,
        #                          robot_name=self.name,
        #                          link_name_tree=LinkNameTree())
        # self.links = self.xml.links
        # self.joints = self.xml.joints


    def set_joint_state(self, joint_name=None, joint_id=None, state=0.0):
        if joint_id is None and joint_name is None:
            raise Exception("Neither joint_name or joint_id is given!!!")
        elif joint_id is None:
            joint_id = 0
            for joint in self.joints:
                if joint_name == joint.name:
                    break
                joint_id += 1

        self.joints[joint_id].set_state(self.link_name_tree, state)

    def render(self, sleep_time=0.1):
        self.vis[self.name].set_transform(
            self.pos.matrix
        )
        for link in self.links:
            link.render()
            time.sleep(sleep_time)


class Urdf_parser():

    def __init__(self,
                 vis,
                 link_name_tree,
                 pos_tree,
                 urdf_path,
                 robot_name
                 ):
        self.urdf_path = urdf_path
        self.vis = vis
        self.link_name_tree = link_name_tree
        self.pos_tree = pos_tree
        self.joints = []
        self.links = []
        self.robot_name = robot_name
        self.robot = self.urdf_parser()
        self.init_base()

    def urdf_parser(self):
        robot = URDF.from_xml_file(os.path.abspath(self.urdf_path))
        return robot

    def init_base(self):
        for link_data in self.robot.links:
            try:
                link_data.collision.geometry
            except AttributeError:
                try:
                    link_data.visual.geometry
                except AttributeError:
                    continue
            init_name = link_data.name
            break
        self.pos_tree = {init_name: Pos()}
        self.link_name_tree.set_base(self.robot_name)
        self.link_name_tree.insert_child(self.robot_name, init_name)

    def identify_geom_type(self, geom):
        if re.search('Box', str(type(geom))) is not None:
            return "box"
        if re.search('Mesh', str(type(geom))) is not None:
            return "mesh"
        if re.search('Cylinder', str(type(geom))) is not None:
            return "cylinder"
        if re.search('Sphere', str(type(geom))) is not None:
            return "sphere"

    def generate_links(self):
        id = 0
        for link_data in self.robot.links:
            id += 1
            name = self.link_name_tree.find_name(link_data.name)
            if name is None:
                continue
            try:
                geom_type = self.identify_geom_type(link_data.collision.geometry)
            except AttributeError:
                try:
                    geom_type = self.identify_geom_type(link_data.visual.geometry)
                except AttributeError:
                    print("link {} has no geometry".format(name))
                    continue
            link_geom = []
            pos = self.pos_tree[name.split("/")[-1]]
            xyz = [0, 0, 0]
            rpy = [0, 0, 0]
            try:
                xyz = link_data.visual.origin.xyz
            except AttributeError:
                print("object {} pos has no xyz".format(link_data.name))
            try:
                rpy = link_data.visual.origin.rpy
            except AttributeError:
                print("object {} pos has no rpy".format(link_data.name))
            visual_pos = {link_data.name + "1": Pos(xyz=xyz, rpy=rpy)}
            if geom_type is not 'mesh':
                try:
                    size = link_data.visual.geometry.size
                except AttributeError:
                    try:
                        size = link_data.collision.geometry.size
                    except AttributeError:
                        continue
                link_geom.append({geom_type: size})
            else:
                try:
                    mesh_path = link_data.collision.geometry.filename
                    link_geom.append({geom_type: mesh_path})
                except AttributeError:
                    print("link {} has no collision".format(name))
                    continue
            link = Link(self.vis, name, id, pos=pos, geom=link_geom, visual_pos=visual_pos)
            self.links.append(link)
        return self.links

    def generate_link_name_tree(self):
        temp_dict = {}
        for joint in self.joints:
            parent_name = joint.parent_link
            child_name = joint.child_link
            name_tree = self.link_name_tree.name_tree
            if re.search(parent_name, str(name_tree)) is not None:
                self.link_name_tree.insert_child(parent_name, child_name)
            else:
                temp_dict.update({parent_name: child_name})
        # the follow code is used if joint is not well queued
        circle_time = 0
        while len(temp_dict) > 0 and circle_time < 10:
            circle_time += 1
            print("the joint is not correctly queued")
            for parent_name in temp_dict:
                print(parent_name)
                print(self.link_name_tree.name_tree)
                name_tree = self.link_name_tree.name_tree
                child_name = temp_dict[parent_name]
                if re.search(parent_name, str(name_tree)) is not None:
                    self.link_name_tree.insert_child(parent_name, child_name)
                    del temp_dict[parent_name]

    def generate_joints(self):
        # TODO have not consider on joint limits
        id = 0
        for joint_data in self.robot.joints:
            id += 1
            name = joint_data.name
            xyz = [i for i in joint_data.origin.xyz]
            pos = Pos(xyz, joint_data.origin.rpy)
            joint_type = joint_data.type
            parent_link = joint_data.parent
            child_link = joint_data.child
            if joint_type is "fixed":
                axis = [0, 0, 0]
            else:
                axis = joint_data.axis
            joint = Joint(self.vis, name, id, pos, joint_type, parent_link, child_link, axis)
            self.joints.append(joint)
            self.pos_tree.update({child_link: pos})
        return self.joints


class Joint:
    def __init__(self,
                 vis,
                 name="Joint1",
                 id=0,
                 pos=Pos(),
                 type="fixed",
                 parent_link="parent_name",
                 child_link="child_name",
                 axis=[0, 0, 0],
                 init_state=0.0,
                 upper=0.0,
                 lower=0.0
                 ):
        # TODO add annotation for this class here
        """

        """
        self.vis = vis
        self.name = name
        self.ID = id
        self.type = type
        self.pos = pos
        self.parent_link = parent_link
        self.child_link = child_link
        self.axis = axis
        self.state = init_state
        self.upper = upper
        self.lower = lower

    def set_name(self, new_name="new_name"):
        self.name = new_name

    def set_id(self, new_id=0):
        self.ID = new_id

    def set_parent(self, new_parent="new_parent"):
        self.parent_link = new_parent

    def set_child(self, new_child="new_child"):
        self.name = new_child

    def set_state(self, name_tree=LinkNameTree(), new_state=0.0):
        self.state = new_state
        name = name_tree.find_name(self.child_link)
        if self.type == "fixed":
            pass
        elif self.type == "revolute":
            print(name)
            self.vis[name].set_transform(
                self.pos.matrix @ tf.rotation_matrix(new_state, self.axis)
            )
        elif self.type == "continuous":
            self.vis[name].set_transform(
                self.pos.matrix @ tf.rotation_matrix(new_state, self.axis)
            )
        elif self.type == "prismatic":
            self.vis[name].set_transform(
                self.pos.matrix @ tf.translation_matrix([i * new_state for i in self.axis])
            )

    def get_state(self):
        # TODO
        if self.type == "fixed":
            pass
        elif self.type == "revolute":
            pass
        elif self.type == "continuous":
            pass
        elif self.type == "prismatic":
            pass

    def render(self):
        # TODO important!!! after we know how to render axis
        pass


class Link:
    def __init__(self,
                 vis,
                 name="base_link",
                 id=0,
                 pos=Pos(),
                 geom=[{"mesh": "meshpath"}, {"box": [0, 0, 0]}],
                 visual_pos={},
                 collision_visible=True,
                 material=None,
                 frame_visible=False):
        # TODO add annotation for this class here
        """
        this class is for create Link in xmirror
        vis is meshcat visualizer
        name is link name
        id is link id
        pos is link pos
        geom is link's geometry can be mesh,box,sphere,cylinder
        """

        self.vis = vis
        self.name = name
        self.ID = id
        self.pos = pos
        self.visual_pos = visual_pos
        self.geom = geom
        self.collision_visible = collision_visible
        self.frame_visible = frame_visible
        self.material = material
        self.visualizer = []


    def init_visualizer(self):
        visual_id = 0
        for geom_item in self.geom:
            geom_type = list(geom_item.keys())[0]
            if geom_type == "mesh":
                self.vis[self.name].set_transform(
                    self.pos.matrix
                )
                mesh_path = geom_item[geom_type]
                name = list(self.visual_pos.keys())[visual_id]
                pos = self.visual_pos[name]
                name = self.name + "/" + name
                self.visualizer.append(visual.Mesh(self.vis, name, mesh_path, pos))
            elif geom_type == "box":
                size = geom_item[geom_type]
                name = list(self.visual_pos.keys())[visual_id]
                pos = self.visual_pos[name]
                name = self.name + "/" + name
                self.visualizer.append(visual.Box(self.vis, name, size=size, material=self.material, pos=pos))
            elif geom_type == "sphere":
                size = geom_item[geom_type]
                name = list(self.visual_pos.keys())[visual_id]
                pos = self.visual_pos[name]
                name = self.name + "/" + name
                self.visualizer.append(visual.Sphere(self.vis, name, size, self.material, pos))
            elif geom_type == "cylinder":
                size = geom_item[geom_type]
                name = list(self.visual_pos.keys())[visual_id]
                pos = self.visual_pos[name]
                name = self.name + "/" + name
                self.visualizer.append(visual.Cylinder(self.vis, name, size, self.material, pos))
            visual_id += 1

    def set_name(self, new_name="new_name"):
        self.name = new_name

    def set_id(self, new_id=0):
        self.ID = new_id

    def view_collision(self, view=True):
        self.collision_visible = view

    def view_frame(self, view=True):
        self.frame_visible = view

    def render(self):
        self.init_visualizer()
        if self.collision_visible:
            for object in self.visualizer:
                object.render()
        if self.frame_visible:
            # TODO
            pass

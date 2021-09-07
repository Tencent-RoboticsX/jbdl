#!/usr/bin/python3

import xml.etree.ElementTree as ET
import time
import numpy as np
from jbdl.experimental.render.xmirror.link_name_tree import LinkNameTree
#import jbdl.experimental.render.xmirror.robot
from jbdl.experimental.render.xmirror.visual import Pos



class World:
    def __init__(self,
                 vis,
                 name,
                 xml_path,
                 mesh_dir,

                 ):
        self.vis = vis
        self.xml_path = xml_path
        self.mesh_dir = mesh_dir
        self.world_name = name

    def load_xml(self):
        self.xml = XmlParser(
            vis=self.vis,
            xml_path=self.xml_path,
            mesh_dir=self.mesh_dir,
            link_name_tree=LinkNameTree())




class XmlParser:
    def __init__(self,
                 vis,
                 xml_path,
                 mesh_dir,
                 link_name_tree=LinkNameTree()
                 ):
        self.xml_path = xml_path
        self.vis = vis
        self.model_name = "model"
        self.link_name_tree = link_name_tree
        self.joints = []
        self.links = []
        self.world = self.load_xml()
        self.meshs = {}
        self.mesh_dir = mesh_dir
        self.models = []
        self.basic_models = []
        self.robot_models = []
        self.__load_meshs()
        self.__load_models()
        self.__basic_model_init()
        self.robot = self.__robot_model_init()
        self.generate_robot()

    def load_xml(self):
        my_tree = ET.parse(self.xml_path)
        root = my_tree.getroot()
        self.model_name = root.get('name')
        return root

    def __load_meshs(self):
        meshs = self.world.findall("asset")[0]
        for mesh in meshs:
            if mesh.tag == "mesh":
                file_path = mesh.get('file')
                mesh_name = file_path[0: file_path.find(".")]
                self.meshs.update({mesh_name: file_path})

    def __load_models(self):
        self.models = self.world.findall("worldbody")[0].findall("body")

    def __basic_model_init(self):
        for model in self.models:
            if len(model.findall('body')) < 1:
                link = self.load_link(model)
                self.basic_models.append(link)

    def __robot_model_init(self):
        for model in self.models:
            if len(model.findall('body')) > 0:
                return model  # TODO quite confused will a xml file include 2 robots?

    def init_base(self):
        init_name = self.robot.get('name')
        if self.robot.get('pos') is not None:
            pos = self.robot.get('pos').split()
        else:
            pos = [0, 0, 0]
        self.pos_tree = {init_name: Pos(xyz=pos)}
        self.link_name_tree.set_base(self.model_name)
        self.link_name_tree.insert_child(self.model_name, init_name)

    def generate_robot(self):
        self.init_base()
        body = self.robot
        while body.findall("body") is not None:
            parent_name = body.get('name')
            link = self.load_link(body)
            link.set_name(self.link_name_tree.find_name(link.name))
            joint = self.load_joint(body)
            self.links.append(link)
            if joint is not None:
                self.joints.append(joint)
            if len(body.findall("body")) > 0:
                body = body.findall("body")[0]
                child_name = body.get('name')
                self.link_name_tree.insert_child(parent_name, child_name)
            else:
                break

    def load_joint(self, model):
        try:
            joint_data = model.findall('joint')[0]
        except IndexError:
            return None
        name = joint_data.get('name')
        link_xyz = [float(i) for i in model.get('pos').split()]
        joint_xyz = [float(i) for i in joint_data.get('pos').split()]
        pos_xyz = np.sum([link_xyz, joint_xyz], axis=0)
        pos = Pos(xyz=pos_xyz)
        if joint_data.get('type') is None:
            joint_type = "revolute"  # TODO default revoulte because there maybe no type
        elif joint_data.get('type') is "hinge":
            joint_type = "revolute"
        elif joint_data.get('type') is "silde":
            joint_type = "prismatic"
        axis = joint_data.get('axis').split()
        child_name = model.get('name')
        parent_name = self.link_name_tree.find_name(child_name).split('/')[-2]
        joint = robot.Joint(self.vis, name, id, pos, joint_type, parent_name, child_name, axis)
        return joint

    def load_link(self, model):
        name = model.get('name')
        pos = Pos(xyz=model.get('pos').split())
        geom_list = []
        visual_pos = {}
        i = 0
        for geom in model.findall("geom"):
            geom_name = geom.get('name')
            if geom_name is None:
                geom_name = name + str(i)
            geom_type = geom.get('type')
            if geom.get('pos') is not None:
                geom_pos = geom.get('pos').split()
            else:
                geom_pos = [0, 0, 0]
            visual_pos.update({geom_name: Pos(xyz=geom_pos)})
            if geom_type == 'mesh':
                mesh_path = self.mesh_dir + "/" + self.meshs[geom.get('mesh')]
                #material = geom.get('material')  # TODO we don't support material for now
                geom_list.append({geom_type: mesh_path})
            else:
                size = geom.get('size').split()
                geom_list.append({geom_type: size})
            i += 1
        link = robot.Link(vis=self.vis,
                          name=name,
                          id=0,
                          pos=pos,
                          geom=geom_list,
                          visual_pos=visual_pos,
                          collision_visible=True,
                          material=None,
                          frame_visible=False)
        return link

    def links_render(self, sleep_time):
        for model in self.links:
            model.render()
            time.sleep(sleep_time)

    def env_render(self, sleep_time):
        # TODO this should be function of world
        for model in self.basic_models:
            model.render()
            time.sleep(sleep_time)

import numpy as np
import json
import jax.numpy as jnp
from jbdl.envs.utils.urdf_reader import URDF
from jbdl.envs.utils.urdf_utils import transform_origin
from jbdl.rbdl.math.spatial_transform import spatial_transform
from jbdl.rbdl.model.rigid_body_inertia import rigid_body_inertia


class UrdfWrapper(object):

    def __init__(self, filepath):
        self.load(filepath)

    @property
    def Xtree(self):
        Xtree = self._model["Xtree"]
        while not isinstance(Xtree[0], np.ndarray):
            Xtree = Xtree[0]
        Xtree = [Xtree[i].copy() for i in range(len(Xtree))]
        return Xtree

    @property
    def inertia(self):
        inertia = self._model["inertia"]
        while not isinstance(inertia[0], np.ndarray):
            inertia = inertia[0]
        inertia = [inertia[i].copy() for i in range(len(inertia))]
        return inertia

    @property
    def a_grav(self):
        a_grav = self._model['a_grav']
        if not isinstance(a_grav, np.ndarray):
            a_grav = np.asfarray(a_grav)
        a_grav = a_grav.flatten().reshape(6, 1)
        return a_grav

    @property
    def parent(self):
        parent = self._model['parent']
        if isinstance(parent, np.ndarray):
            parent = self._model['parent'].flatten().astype(int).tolist()
        return parent

    @property
    def NB(self):
        NB = int(self._model["NB"])
        return NB

    @property
    def jtype(self):
        jtype = self._model["jtype"]
        if isinstance(jtype, np.ndarray):
            jtype = jtype.flatten().astype(int).tolist()
        return jtype

    @property
    def jaxis(self):
        jaxis = self._model['jaxis']
        return jaxis

    @property
    def model(self):
        model = dict()
        # model["NB"] = self.NB
        model["nb"] = self.NB
        model["a_grav"] = self.a_grav
        model["jtype"] = self.jtype
        model["jaxis"] = self.jaxis
        # model["Xtree"] = self.Xtree
        model["x_tree"] = self.Xtree
        # model["I"] = self.I
        model["inertia"] = self.inertia
        model["parent"] = self.parent
        model["jname"] = self.jname
        model["urdf_path"] = self.urdf_path
        model["NC"] = self.NC
        model["idcontact"] = self.idcontact
        model["contactpoint"] = self.contactpoint
        return model

    @property
    def json(self):
        json_model = dict()
        for key, value in self.model.items():
            if isinstance(value, np.ndarray):
                json_model[key] = value.tolist()
            elif isinstance(value, list):
                json_list = []
                for elem in value:
                    if isinstance(elem, np.ndarray):
                        json_list.append(elem.tolist())
                    else:
                        json_list.append(elem)
                json_model[key] = json_list
            else:
                json_model[key] = value
        return json_model

    @property
    def jname(self):
        jname = self._model['jname']
        return jname

    @property
    def urdf_path(self):
        urdf_path = self._model['urdf_path']
        return urdf_path

    @property
    def NC(self):
        nc = self._model['NC']
        return nc

    @property
    def idcontact(self):
        idcontact = self._model['idcontact']
        return idcontact

    @property
    def contactpoint(self):
        contactpoint = self._model['contactpoint']
        return contactpoint

    def save(self, file_path: str):
        with open(file_path, 'w') as outfile:
            json.dump(self.json, outfile, indent=2)

    def load(self, file_path: str):
        urdf_model = load_urdf(file_path)
        self._model = urdf_model


def load_urdf(file_path):
    robot = URDF.load(file_path)  # "/root/Downloads/urdf/arm.urdf"
    model = dict()

    # NB
    NB = len(robot.links)
    model["NB"] = NB

    # NC
    NC = 0
    _contact_id = []
    _contact_points = []
    # for i in range(NB):
    #    _numcols = len(robot.links[i].collisions)
    #    if(_numcols>0):
    #        NC+=1
    #        _contact_id.append(i+1) #bodyid plus 1
    # _contact_points.append(np.zeros((3,)))#assume not contact at all
    # [0.0,-0.30,0.0]
    for i in range(NB):
        _numcols = len(robot.links[i].collisions)
        _name = robot.links[i].name
        if(_numcols > 0):
            if(_name.endswith('lower_leg')):
                NC += 1
                _contact_id.append(i + 1)  # bodyid plus 1
                # assume not contact at all [0.0,-0.30,0.0]
                _contact_points.append(np.array([0.0, -0.280, 0.0]))
    model['NC'] = NC
    model['idcontact'] = _contact_id.copy()
    model['contactpoint'] = _contact_points.copy()

    # joint_name
    joint_name = []
    joint_name.append("")  # the first joint does't not exist
    for i in range(NB - 1):
        name = robot.joints[i].name
        joint_name.append(name)
    model['jname'] = joint_name

    # urdf
    model['urdf_path'] = file_path

    # grav
    a_grav = np.zeros((6, 1))
    a_grav[5] = -9.81
    model["a_grav"] = a_grav

    # jtype
    joint_type = [0] * NB
    joint_type[0] = 1  # TODO: need discuss
    for i in range(NB - 1):
        if(robot.joints[i].joint_type == "revolute"):
            joint_type[i + 1] = 0
        elif(robot.joints[i].joint_type == "continuous"):
            joint_type[i + 1] = 0
        elif(robot.joints[i].joint_type == "prismatic"):
            joint_type[i + 1] = 1
        else:
            joint_type[i + 1] = 1  # TODO need discuss
            # print("not known joint type",i)
    model['jtype'] = joint_type

    # joint_axis
    joint_axis = ""
    joint_axis += 'z'  # TODO:need discuss x,y for fixed base, z for moving base
    for i in range(NB - 1):
        axis_type = robot.joints[i].axis
        if(axis_type[0] == 1):
            joint_axis += 'x'
        elif(axis_type[1] == 1):
            joint_axis += 'y'
        elif(axis_type[2] == 1):
            joint_axis += 'z'
        elif(axis_type[0] == -1):
            joint_axis += 'a'
        elif(axis_type[1] == -1):
            joint_axis += 'b'
        elif(axis_type[2] == -1):
            joint_axis += 'c'
        else:
            joint_axis += 'x'
            # print("no known joint axis",i)
    model['jaxis'] = joint_axis  # joint_axis#

    # parents
    parents = [0] * NB
    parents[0] = 0  # base link
    name_dict = {}
    for i in range(NB):
        _n = robot.links[i].name
        name_dict[_n] = i
    for i in range(NB - 1):
        _p, _c = robot.joints[i].parent, robot.joints[i].child
        _pi, _ci = name_dict[_p], name_dict[_c]
        # TODO this is not suggested, but to sync with pyRBDL
        parents[_ci] = _pi + 1
    model['parent'] = parents

    # xtree and I
    X_tree, inertia = [], []
    for i in range(len(robot.links)):
        if(i == 0):
            trans_matrix = np.zeros((3,))
            rot_matrix = np.eye(3)
        else:
            xyz, rpy = transform_origin(robot.joints[i - 1].origin)
            origin_matrix = robot.joints[i - 1].origin
            trans_matrix = origin_matrix[:3, 3]  # xyz
            rot_matrix = origin_matrix[:3, :3]  # rpy_to_matrix(rpy)
        link_mass = robot.links[i].inertial.mass
        link_intertia = robot.links[i].inertial.inertia
        link_com = transform_origin(robot.links[i].inertial.origin)[
            0]  # defaul rpy in link is equal to 0,0,0
        # tranform
        tree_element = spatial_transform(
            jnp.asarray(rot_matrix),
            jnp.asarray(trans_matrix))
        I_element = rigid_body_inertia(
            link_mass,
            jnp.asarray(link_com),
            jnp.asarray(link_intertia))
        # build tree
        X_tree.append(np.asarray(tree_element))
        inertia.append(np.asarray(I_element))
    model['Xtree'] = X_tree
    model['inertia'] = inertia

    return model

# x = UrdfWrapper()
# x.load("/root/Downloads/urdf/arm.urdf")

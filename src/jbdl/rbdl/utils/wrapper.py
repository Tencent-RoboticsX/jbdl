import json
import numpy as np


def jsonize(py_dict: dict) -> dict:
    json_model = dict()
    for key, value in py_dict.items():
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
        elif isinstance(value, dict):
            json_model[key] = jsonize(value)
        else:
            json_model[key] = value
    return json_model


class ModelWrapper(object):

    def __init__(self, model=None):
        if model is None:
            self._model = dict()
        else:
            self._model = model

    @property
    def x_tree(self):
        x_tree = self._model["x_tree"]
        while type(x_tree[0]) is not np.ndarray:
            x_tree = x_tree[0]
        x_tree = [x_tree[i].copy() for i in range(len(x_tree))]
        return x_tree


    @property
    def inertia(self):
        inertia = self._model["inertia"]
        while type(inertia[0]) is not np.ndarray:
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
    def idcomplot(self):
        idcomplot = self._model["idcomplot"]
        if isinstance(idcomplot, np.ndarray):
            idcomplot = idcomplot.flatten().astype(int).tolist()
        return idcomplot



    @property
    def idlinkplot(self):
        idlinkplot = self._model["idlinkplot"]
        if isinstance(idlinkplot, np.ndarray):
            idlinkplot = idlinkplot.flatten().astype(int).tolist()
        return idlinkplot

    @idlinkplot.setter
    def idlinkplot(self, idlinkplot):
        self._model["idlinkplot"] = idlinkplot

    @property
    def linkplot(self):
        linkplot = self._model["linkplot"]
        while type(linkplot[0]) is not np.ndarray:
            linkplot = linkplot[0]
        linkplot = [linkplot[i].copy() for i in range(len(linkplot))]
        return linkplot

    @linkplot.setter
    def linkplot(self, linkplot):
        self._model["linkplot"] = linkplot

    @property
    def idcontact(self):
        idcontact = self._model['idcontact']
        if isinstance(idcontact, np.ndarray):
            idcontact = idcontact.flatten().astype(int).tolist()
        return idcontact

    @idcontact.setter
    def idcontact(self, idcontact):
        self._model["idcontact"] = idcontact

    @property
    def contactpoint(self):
        contactpoint = self._model["contactpoint"]
        while type(contactpoint[0]) is not np.ndarray:
            contactpoint = contactpoint[0]
        contactpoint = [contactpoint[i].copy() for i in range(len(contactpoint))]
        return contactpoint

    @contactpoint.setter
    def contactpoint(self, contactpoint):
        self._model["contactpoint"] = contactpoint

    @property
    def parent(self):
        parent = self._model['parent']
        if isinstance(parent, np.ndarray):
            parent = parent.flatten().astype(int).tolist()
        return parent

    @property
    def nb(self):
        nb = int(self._model["nb"])
        return nb

    @nb.setter
    def nb(self, nb):
        self._model["nb"] = nb

    @property
    def nf(self):
        nf = int(self._model["nf"])
        return nf

    @nf.setter
    def nf(self, nf):
        self._model["nf"] = nf

    @property
    def contact_force_lb(self):
        contact_force_lb = np.array(self._model["contact_cond"]["contact_force_lb"]).reshape(*(-1, 1))
        return contact_force_lb

    @contact_force_lb.setter
    def contact_force_lb(self, contact_force_lb: np.ndarray):
        if self._model.get("contact_cond") is None:
            self._model["contact_cond"] = dict()
        self._model["contact_cond"]["contact_force_lb"] = contact_force_lb.reshape(*(-1, 1))

    @property
    def contact_force_ub(self):
        contact_force_ub = np.array(self._model["contact_cond"]["contact_force_ub"]).reshape(*(-1, 1))
        return contact_force_ub

    @contact_force_ub.setter
    def contact_force_ub(self, contact_force_ub: np.ndarray):
        if self._model.get("contact_cond") is None:
            self._model["contact_cond"] = dict()
        self._model["contact_cond"]["contact_force_ub"] = contact_force_ub.reshape(*(-1, 1))

    @property
    def contact_force_kp(self):
        contact_force_kp = np.array(self._model["contact_cond"]["contact_force_kp"]).reshape(*(-1, 1))
        return contact_force_kp

    @contact_force_kp.setter
    def contact_force_kp(self, contact_force_kp: np.ndarray):
        if self._model.get("contact_cond") is None:
            self._model["contact_cond"] = dict()
        self._model["contact_cond"]["contact_force_kp"] = contact_force_kp.reshape(*(-1, 1))

    @property
    def contact_force_kd(self):
        contact_force_kd = np.array(self._model["contact_cond"]["contact_force_kd"]).reshape(*(-1, 1))
        return contact_force_kd

    @contact_force_kd.setter
    def contact_force_kd(self, contact_force_kd: np.ndarray):
        if self._model.get("contact_cond") is None:
            self._model["contact_cond"] = dict()
        self._model["contact_cond"]["contact_force_kd"] = contact_force_kd.reshape(*(-1, 1))

    @property
    def contact_pos_lb(self):
        contact_pos_lb = np.array(self._model["contact_cond"]["contact_pos_lb"]).reshape(*(-1, 1))
        return contact_pos_lb

    @contact_pos_lb.setter
    def contact_pos_lb(self, contact_pos_lb: np.ndarray):
        if self._model.get("contact_cond") is None:
            self._model["contact_cond"] = dict()
        self._model["contact_cond"]["contact_pos_lb"] = contact_pos_lb.reshape(*(-1, 1))

    @property
    def contact_pos_ub(self):
        contact_pos_ub = np.array(self._model["contact_cond"]["contact_pos_ub"]).reshape(*(-1, 1))
        return contact_pos_ub

    @contact_pos_ub.setter
    def contact_pos_ub(self, contact_pos_ub: np.ndarray):
        if self._model.get("contact_cond") is None:
            self._model["contact_cond"] = dict()
        self._model["contact_cond"]["contact_pos_ub"] = contact_pos_ub.reshape(*(-1, 1))

    @property
    def contact_vel_lb(self):
        contact_vel_lb = np.array(self._model["contact_cond"]["contact_vel_lb"]).reshape(*(-1, 1))
        return contact_vel_lb

    @contact_vel_lb.setter
    def contact_vel_lb(self, contact_vel_lb: np.ndarray):
        if self._model.get("contact_cond") is None:
            self._model["contact_cond"] = dict()
        self._model["contact_cond"]["contact_vel_lb"] = contact_vel_lb.reshape(*(-1, 1))

    @property
    def contact_vel_ub(self):
        contact_vel_ub = np.array(self._model["contact_cond"]["contact_vel_ub"]).reshape(*(-1, 1))
        return contact_vel_ub

    @contact_vel_ub.setter
    def contact_vel_ub(self, contact_vel_ub: np.ndarray):
        if self._model.get("contact_cond") is None:
            self._model["contact_cond"] = dict()
        self._model["contact_cond"]["contact_vel_ub"] = contact_vel_ub.reshape(*(-1, 1))

    @property
    def contact_cond(self):
        contact_cond = self._model.get("contact_cond", None)
        return contact_cond

    @contact_cond.setter
    def contact_cond(self, contact_cond: dict):
        contact_force_lb = contact_cond.get("contact_force_lb")
        contact_force_ub = contact_cond.get("contact_force_ub")
        contact_force_kp = contact_cond.get("contact_force_kp")
        contact_force_kd = contact_cond.get("contact_force_kd")
        if contact_force_lb is not None:
            self.contact_force_lb = contact_force_lb
        if contact_force_ub is not None:
            self.contact_force_ub = contact_force_ub
        if contact_force_kp is not None:
            self.contact_force_kp = contact_force_kp
        if contact_force_kd is not None:
            self.contact_force_kd = contact_force_kd

    @property
    def nc(self):
        nc = int(self._model["nc"])
        return nc

    @nc.setter
    def nc(self, nc):
        self._model["nc"] = nc

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
    def com(self):
        com = self._model["com"]
        while type(com[0]) is not np.ndarray:
            com = com[0]
        com = [com[i].copy() for i in range(len(com))]
        return com

    @com.setter
    def com(self, com):
        self._model["com"] = com

    @property
    def ic(self):
        ic = self._model["ic"]
        while type(ic[0]) is not np.ndarray:
            ic = ic[0]
        ic = [ic[i].copy() for i in range(len(ic))]
        return ic

    @ic.setter
    def ic(self, ic):
        self._model["ic"] = ic

    @property
    def st(self):
        st = self._model["st"]
        while type(st[0]) is not np.ndarray:
            st = st[0]
        return st

    @st.setter
    def st(self, st):
        self._model["st"] = st

    @property
    def mass(self):
        mass = self._model['mass']
        if isinstance(mass, np.ndarray):
            mass = mass.flatten().tolist()
        return mass

    @mass.setter
    def mass(self, mass):
        self._model["mass"] = mass

    @property
    def model(self):
        model = dict()
        model["nb"] = self.nb
        model["nc"] = self.nc
        model["nf"] = self.nf
        model["a_grav"] = self.a_grav
        model["jtype"] = self.jtype
        model["jaxis"] = self.jaxis
        model["x_tree"] = self.x_tree
        model["inertia"] = self.inertia
        model["parent"] = self.parent
        model["idcomplot"] = self.idcomplot
        model["idlinkplot"] = self.idlinkplot
        model["idcontact"] = self.idcontact
        model["contactpoint"] = self.contactpoint
        model["com"] = self.com
        model["linkplot"] = self.linkplot
        model["ic"] = self.ic
        model["mass"] = self.mass
        model["st"] = self.st
        model["contact_cond"] = self.contact_cond
        return model

    @property
    def json(self):
        json_model = jsonize(self.model)
        return json_model

    def save(self, file_path: str):
        with open(file_path, 'w') as outfile:
            json.dump(self.json, outfile, indent=2)

    def load(self, file_path: str):
        with open(file_path, 'r') as infile:
            json_model = json.load(infile)

        model = dict()
        for key, value in json_model.items():
            model[key] = value
            if key in ["x_tree", "inertia", "com", "linkplot", "contactpoint", "ic"]:
                model[key] = [np.asarray(elem) for elem in value]
            elif key in ["a_grav", "mass", "st"]:
                model[key] = np.asfarray(value)
            elif key in ["contact_cond"]:
                sub_model = dict()
                for sub_key, sub_value in value.items():
                    sub_model[sub_key] = np.asfarray(sub_value)
                model[key] = sub_model
            else:
                model[key] = value
        self._model = model

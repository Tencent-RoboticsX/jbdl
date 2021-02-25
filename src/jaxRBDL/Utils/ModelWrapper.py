import numpy as np
import json

class ModelWrapper(object):

    def __init__(self, model):
        self._model = model

    @property
    def Xtree(self):
        Xtree = self._model["Xtree"]
        while type(Xtree[0]) is not np.ndarray:
            Xtree = Xtree[0]
        Xtree = [Xtree[i].copy() for i in range(len(Xtree))]
        return Xtree


    @property
    def I(self):
        I = self._model["I"]
        while type(I[0]) is not np.ndarray:
            I = I[0]
        I = [I[i].copy() for i in range(len(I))]
        return I

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
        model["NB"] = self.NB
        model["jtype"] = self.jtype
        model["jaxis"] = self.jaxis
        model["Xtree"] = self.Xtree
        model["I"] = self.I
        model["parent"] = self.parent
        return model

    @property
    def json(self):
        json_model = dict()
        for key, value in self.model.items():
            if isinstance(value, list):
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

    def save(self, file_path: str):
        with open(file_path, 'w') as outfile:
            json.dump(self.json, outfile, indent=2)

    def load(self, file_path: str):
        with open(file_path, 'r') as infile:
            json_model = json.load(infile)

        model = dict() 
        for key, value in json_model.items():
            model[key] = value
            if key in ["Xtree", "I"]:
                model[key] = [np.asarray(elem) for elem in value]
            else:
                model[key] = value
        self._model = model




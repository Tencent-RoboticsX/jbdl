import re


class LinkNameTree:
    def __init__(self, base_name="base"):
        self.name_tree = {base_name: {}}

    def set_base(self, name):
        self.name_tree = {name: {}}

    def find_name(self,
                  target="",
                  dic=None,
                  current_name=""):
        if dic is None:
            dic = self.name_tree
        if re.search(target, str(dic)) is not None:
            if target in dic.keys():
                pos = current_name + target
                return pos
            else:
                for key in dic.keys():
                    current_pos = current_name + key + "/"
                    return self.find_name(target, dic[key], current_pos)
        else:
            print("there is no link_name called " + target)

    def get_index(self, name: str = ""):
        index = name.split("/")
        return index

    def get_child(self, name=""):
        index = self.get_index(name)
        print(index)
        temp = self.name_tree
        for each in index:
            temp = temp[each]
        return temp

    def insert_child(self, parent_link="", link_name=""):
        parent_name = self.find_name(parent_link)
        parent_index = self.get_index(parent_name)
        temp = self.name_tree
        for each in parent_index:
            temp = temp[each]
        temp.update({link_name: {}})

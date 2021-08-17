
def xyz2int(jtype: str):
    return tuple(['xyz'.index(s) for s in jtype])


if __name__ == '__main__':
    print(xyz2int("xyzzz"))



from collections import deque

JTYPE_LIST = ['x', 'y', 'z', '-x', '-y', '-z']

def xyz2int(jtype: str):
    q = deque(jtype)
    res = []
    while q:
        elem = q.popleft()
        if elem != '-':
            res.append(JTYPE_LIST.index(elem))
        else:
            
            res.append(JTYPE_LIST.index('-' + q.popleft()))
    return tuple(res)
    


# def xyz2int(jtype: str):
#     return tuple(['xyz'.index(s) for s in jtype])


if __name__ == '__main__':
    print(xyz2int("xyzzz"))
    # print(xyz_to_int("x-yzz-z"))

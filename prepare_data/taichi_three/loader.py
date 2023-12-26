import numpy as np



def _append(faces, indices):
    if len(indices) == 4:
        faces.append([indices[0], indices[1], indices[2]])
        faces.append([indices[2], indices[3], indices[0]])
    elif len(indices) == 3:
        faces.append(indices)
    else:
        assert False, len(indices)


def readobj(path, scale=1):
    vi = []
    vt = []
    vn = []
    faces = []

    with open(path, 'r') as myfile:
        lines = myfile.readlines()

    # cache vertices
    for line in lines:
        try:
            type, fields = line.split(maxsplit=1)
            fields = [float(_) for _ in fields.split()]
        except ValueError:
            continue

        if type == 'v':
            vi.append(fields)
        elif type == 'vt':
            vt.append(fields)
        elif type == 'vn':
            vn.append(fields)

    # cache faces
    for line in lines:
        try:
            type, fields = line.split(maxsplit=1)
            fields = fields.split()
        except ValueError:
            continue

        # line looks like 'f 5/1/1 1/2/1 4/3/1'
        # or 'f 314/380/494 382/400/494 388/550/494 506/551/494' for quads
        if type != 'f':
            continue

        # a field should look like '5/1/1'
        # for vertex/vertex UV coords/vertex Normal  (indexes number in the list)
        # the index in 'f 5/1/1 1/2/1 4/3/1' STARTS AT 1 !!!
        
        indices = [[int(_) - 1 if _ != '' else 0 for _ in field.split('/')] for field in fields]

        if len(indices) == 4:
            faces.append([indices[0], indices[1], indices[2]])
            faces.append([indices[2], indices[3], indices[0]])
        elif len(indices) == 3:
            faces.append(indices)
        else:
            assert False, len(indices)

    ret = {}
    ret['vi'] = None if len(vi) == 0 else np.array(vi).astype(np.float32) * scale
    ret['vt'] = None if len(vt) == 0 else np.array(vt).astype(np.float32)
    ret['vn'] = None if len(vn) == 0 else np.array(vn).astype(np.float32)
    ret['f'] = None if len(faces) == 0 else np.array(faces).astype(np.int32)
    return ret

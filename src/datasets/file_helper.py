from .. import util
import numpy as np

def read_array_file(file, type):
    data = util.readLines(file)
    data = [x.split() for x in data]
    data = np.array(data).astype(type)

    return data

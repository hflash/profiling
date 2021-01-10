import os


def dir_mk(name):
    dirname = name + '_slices'
    dir = os.getcwd()
    print(os.getcwd())
    folder = os.path.exists(dirname)
    if not folder:
        os.makedirs(dir + dirname)


def qasm_create(name, i):
    path = name
    dirname = './' + path + '_slices'
    if os.path.exists(dirname):
        pass
    else:
        os.mkdir(dirname)
    filename = './' + path + '_slices' + str(i) + '.qasm'
    f = open(filename, 'x')
    f.close()


if __name__ == 'main':
    path = 'qft'
    dir_mk(path)
    # for i in range(5):
    #     qasm_create(path, i)


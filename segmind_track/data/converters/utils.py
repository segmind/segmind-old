import os
import xml.etree.ElementTree as ET
from os import listdir
from os.path import basename, dirname, isfile, join, splitext


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError(
            'The size of %s is supposed to be %d, but is %d.' %
            (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename(filename):
    filename = os.path.splitext(filename)
    return (filename)


class XMLReader:

    def __init__(self, path, output_dir):
        file = open(path, 'r')

        self.path = path
        self.output_dir = output_dir
        self.content = file.read()
        self.root = ET.fromstring(self.content)
        self.template = '{name} 0.00 0 0.0 {xmin} {ymin} {xmax} {ymax}\
         0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0'

    def get_filename(self):
        return splitext(basename(self.path))[0]

    def get_dir(self):
        return dirname(self.path)

    def get_objects(self):
        objects = []

        for object in self.root.findall('object'):
            objects.append({
                'name': object.find('name').text,
                'xmin': object.find('bndbox').find('xmin').text,
                'ymin': object.find('bndbox').find('ymin').text,
                'xmax': object.find('bndbox').find('xmax').text,
                'ymax': object.find('bndbox').find('ymax').text
            })

        return objects

    def fill_template(self, object):
        return self.template.format(**object)

    def export_kitti(self):
        objects = self.get_objects()

        # Skip empty
        if len(objects) == 0:
            return False

        file = open(join(self.output_dir, self.get_filename()) + '.txt', 'w')

        for object in objects[:-1]:
            file.write(self.fill_template(object) + '\n')
        # Write last without '\n'
        file.write(self.fill_template(objects[-1]))

        file.close()

        return True


def process_file(path, output_dir):
    xml_reader = XMLReader(path, output_dir)

    return xml_reader.export_kitti()


def get_directory_xml_files(dir):
    return [
        join(dir, f) for f in listdir(dir)
        if isfile(join(dir, f)) and splitext(f)[1].lower() == '.xml'
    ]


def check_argv(argv):
    return len(argv) > 1

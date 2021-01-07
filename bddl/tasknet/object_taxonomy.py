import networkx as nx
from IPython import embed
import json


class ObjectTaxonomy(object):
    def __init__(self, json_file):
        self.taxonomy = self.parse_taxonomy(json_file)

    def parse_taxonomy(self, json_file):
        with open(json_file) as f:
            json_obj = json.load(f)

        DG = nx.DiGraph()
        nodes = [json_obj]
        while len(nodes) > 0:
            next_nodes = []
            for node in nodes:
                DG.add_node(node['name'],
                            properties=set(['cookable', 'dustable']))
                if 'children' in node:
                    for child in node['children']:
                        next_nodes.append(child)
                        DG.add_edge(node['name'], child['name'])
            nodes = next_nodes
        return DG

    def is_valid(self, object_name):
        return self.taxonomy.has_node(object_name)

    def descendants(self, object_name):
        assert self.is_valid(object_name)
        return nx.algorithms.dag.descendants(self.taxonomy, object_name)

    def leaf_descendants(self, object_name):
        return list(filter(lambda node: self.taxonomy.out_degree(node) == 0,
                           self.descendants(object_name)))

    def ancestors(self, object_name):
        assert self.is_valid(object_name)
        return nx.algorithms.dag.ancestors(self.taxonomy, object_name)

    def is_desendent(self, object_name_a, object_name_b):
        assert self.is_valid(object_name_a)
        assert self.is_valid(object_name_b)
        return object_name_a in self.descendants(object_name_b)

    def is_ancestor(self, object_name_a, object_name_b):
        assert self.is_valid(object_name_a)
        assert self.is_valid(object_name_b)
        return object_name_a in self.ancestors(object_name_b)

    def properties(self, object_name):
        assert self.is_valid(object_name)
        return self.taxonomy.nodes[object_name]['properties']

    def has_property(self, object_name, property_name):
        return property_name in self.properties(object_name)


if __name__ == "__main__":
    object_taxonomy = ObjectTaxonomy('test_taxonomy.json')

import copy
import json
import pkgutil

import networkx as nx
from IPython import embed
import bddl

DEFAULT_HIERARCHY_FILE = pkgutil.get_data(
    bddl.__package__, 'hierarchy_owned.json')


class ObjectTaxonomy(object):
    def __init__(self, hierarchy_type="owned"):
        hierarchy_file = pkgutil.get_data(bddl.__package__, f"hierarchy_{hierarchy_type}.json")
        self.taxonomy = self._parse_taxonomy(hierarchy_file)

    @staticmethod
    def _parse_taxonomy(json_str):
        """
        Parse taxonomy from hierarchy JSON file.

        :param json_str: str containing JSON-encoded hierarchy.
        :return: networkx.DiGraph corresponding to object hierarchy tree with classes as nodes and parent-child
            relationships as directed edges from parent to child.
        """
        json_obj = json.loads(json_str)

        taxonomy = nx.DiGraph()
        nodes = [(json_obj, None)]
        while len(nodes) > 0:
            next_nodes = []
            for node, parent in nodes:
                children_names = set()
                if 'children' in node:
                    for child in node['children']:
                        next_nodes.append((child, node))
                        children_names.add(child['name'])
                taxonomy.add_node(node['name'],
                                  igibson_categories=node['igibson_categories'],
                                  abilities=node['abilities'])
                for child_name in children_names:
                    taxonomy.add_edge(node['name'], child_name)
            nodes = next_nodes
        return taxonomy

    def _get_class_by_filter(self, filter_fn):
        """
        Gets a single class matching the given filter function.

        :param filter_fn: Filter function that takes a single class name as input and returns a boolean (True to keep).
        :return: str corresponding to the matching class name, None if no match found.
        :raises: ValueError if more than one matching class is found.
        """
        matched = [
            class_name for class_name in self.taxonomy.nodes if filter_fn(class_name)]

        if not matched:
            return None
        elif len(matched) > 1:
            raise ValueError("Multiple classes matched: %s" %
                             ", ".join(matched))

        return matched[0]

    def get_class_name_from_igibson_category(self, igibson_category):
        """
        Get class name corresponding to iGibson object category.

        :param igibson_category: iGibson object category to search for.
        :return: str containing matching class name.
        :raises ValueError if multiple matching classes are found.
        """
        return self._get_class_by_filter(lambda class_name: igibson_category in self.get_igibson_categories(class_name))

    def get_subtree_igibson_categories(self, class_name):
        """
        Get the iGibson object categories matching the subtree of a given class (by aggregating categories across all the leaf-level descendants).

        :param class name: Class name to search
        :return: list of str corresponding to iGibson object categories
        """
        if self.is_leaf(class_name):
            class_names = [class_name]
        else:
            class_names = self.get_leaf_descendants(class_name)
        all_igibson_categories = []
        for class_name in class_names:
            all_igibson_categories += self.get_igibson_categories(class_name)
        return all_igibson_categories

    def is_valid_class(self, class_name):
        """
        Check whether a given class exists within the object taxonomy.

        :param class_name: Class name to search.
        :return: bool indicating if the class exists in the taxonomy.
        """
        return self.taxonomy.has_node(class_name)

    def get_descendants(self, class_name):
        """
        Get the descendant classes of a class.

        :param class_name: Class name to search.
        :return: list of str corresponding to descendant class names.
        """
        assert self.is_valid_class(class_name)
        return list(nx.algorithms.dag.descendants(self.taxonomy, class_name))

    def get_leaf_descendants(self, class_name):
        """
        Get the leaf descendant classes of a class, e.g. descendant classes who are also leaf nodes.

        :param class_name: Class name to search.
        :return: list of str corresponding to leaf descendant class names.
        """
        return [node for node in self.get_descendants(class_name) if self.taxonomy.out_degree(node) == 0]

    def get_ancestors(self, class_name):
        """
        Get the ancestor classes of a class.

        :param class_name: Class name to search.
        :return: list of str corresponding to ancestor class names.
        """
        assert self.is_valid_class(class_name)
        return list(nx.algorithms.dag.ancestors(self.taxonomy, class_name))

    def is_descendant(self, class_name, potential_ancestor_class_name):
        """
        Check whether a given class is a descendant of another class.

        :param class_name: The class name being searched for as a descendant.
        :param potential_ancestor_class_name: The class name being searched for as a parent.
        :return: bool indicating whether class_name is a descendant of potential_ancestor_class_name.
        """
        assert self.is_valid_class(class_name)
        assert self.is_valid_class(potential_ancestor_class_name)
        return class_name in self.get_descendants(potential_ancestor_class_name)

    def is_ancestor(self, class_name, potential_descendant_class_name):
        """
        Check whether a given class is an ancestor of another class.

        :param class_name: The class name being searched for as an ancestor.
        :param potential_descendant_class_name: The class name being searched for as a descendant.
        :return: bool indicating whether class_name is an ancestor of potential_descendant_class_name.
        """
        assert self.is_valid_class(class_name)
        assert self.is_valid_class(potential_descendant_class_name)
        return class_name in self.get_ancestors(potential_descendant_class_name)

    def get_abilities(self, class_name):
        """
        Get the abilities of a given class.

        :param class_name: Class name to search.
        :return: dict in the form of {ability: {param: value}} containing abilities and ability parameters.
        """
        assert self.is_valid_class(class_name), f"Invalid class name: {class_name}"
        return copy.deepcopy(self.taxonomy.nodes[class_name]['abilities'])

    def get_igibson_categories(self, class_name):
        """
        Get the iGibson object categories matching a given class.

        :param class_name: Class name to search.
        :return: list of str corresponding to iGibson object categories matching the class.
        """
        assert self.is_valid_class(class_name)
        return list(self.taxonomy.nodes[class_name]['igibson_categories'])

    def get_children(self, class_name):
        """
        Get the immediate child classes of a class.

        :param class_name: Class name to search.
        :return: list of str corresponding to child class names.
        """
        assert self.is_valid_class(class_name)
        return list(self.taxonomy.successors(class_name))

    def get_parent(self, class_name):
        """
        Get the immediate parent class of a class.

        :param class_name: Class name to search.
        :return: str corresponding to parent class name, None if no parent exists.
        """
        assert self.is_valid_class(class_name)

        in_degree = self.taxonomy.in_degree(class_name)
        assert in_degree <= 1

        return next(self.taxonomy.predecessors(class_name)) if in_degree else None

    def is_leaf(self, class_name):
        """
        Check whether a given class is a leaf class e.g. it has no descendants.

        :param class_name: Class name to search.
        :return: bool indicating if the class is a leaf class.
        """
        assert self.is_valid_class(class_name), "{} is not a valid class name".format(class_name)
        return self.taxonomy.out_degree(class_name) == 0

    def has_ability(self, class_name, ability):
        """
        Check whether the given class has the given ability.

        :param class_name: Class name to check.
        :param ability: Ability name to check.
        :return: bool indicating if the class has the ability.
        """
        return ability in self.get_abilities(class_name)


if __name__ == "__main__":
    object_taxonomy = ObjectTaxonomy()
    embed()

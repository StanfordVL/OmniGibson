import copy
import json
import pkgutil

import networkx as nx
from IPython import embed
import bddl

DEFAULT_HIERARCHY_FILE = pkgutil.get_data(
    bddl.__package__, "generated_data/output_hierarchy_properties.json")


class ObjectTaxonomy(object):
    def __init__(self, hierarchy_type="default"):
        hierarchy_file = DEFAULT_HIERARCHY_FILE 
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
                                  categories=node.get('categories', set()),
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
            synset for synset in self.taxonomy.nodes if filter_fn(synset)]

        if not matched:
            return None
        elif len(matched) > 1:
            raise ValueError("Multiple classes matched: %s" %
                             ", ".join(matched))

        return matched[0]

    def get_synset_from_igibson_category(self, igibson_category):
        """
        Get class name corresponding to iGibson object category.

        :param igibson_category: iGibson object category to search for.
        :return: str containing matching class name.
        :raises ValueError if multiple matching classes are found.
        """
        return self._get_class_by_filter(lambda synset: igibson_category in self.get_categories(synset))

    def get_subtree_categories(self, synset):
        """
        Get the iGibson object categories matching the subtree of a given class (by aggregating categories across all the leaf-level descendants).

        :param class name: Class name to search
        :return: list of str corresponding to iGibson object categories
        """
        if self.is_leaf(synset):
            synsets = [synset]
        else:
            synsets = self.get_leaf_descendants(synset)
        all_categories = []
        for synset in synsets:
            all_categories += self.get_categories(synset)
        return all_categories

    def is_valid_class(self, synset):
        """
        Check whether a given class exists within the object taxonomy.

        :param synset: Class name to search.
        :return: bool indicating if the class exists in the taxonomy.
        """
        return self.taxonomy.has_node(synset)

    def get_descendants(self, synset):
        """
        Get the descendant classes of a class.

        :param synset: Class name to search.
        :return: list of str corresponding to descendant class names.
        """
        assert self.is_valid_class(synset)
        return list(nx.algorithms.dag.descendants(self.taxonomy, synset))

    def get_leaf_descendants(self, synset):
        """
        Get the leaf descendant classes of a class, e.g. descendant classes who are also leaf nodes.

        :param synset: Class name to search.
        :return: list of str corresponding to leaf descendant class names.
        """
        return [node for node in self.get_descendants(synset) if self.taxonomy.out_degree(node) == 0]

    def get_ancestors(self, synset):
        """
        Get the ancestor classes of a class.

        :param synset: Class name to search.
        :return: list of str corresponding to ancestor class names.
        """
        assert self.is_valid_class(synset)
        return list(nx.algorithms.dag.ancestors(self.taxonomy, synset))

    def is_descendant(self, synset, potential_ancestor_synset):
        """
        Check whether a given class is a descendant of another class.

        :param synset: The class name being searched for as a descendant.
        :param potential_ancestor_synset: The class name being searched for as a parent.
        :return: bool indicating whether synset is a descendant of potential_ancestor_synset.
        """
        assert self.is_valid_class(synset)
        assert self.is_valid_class(potential_ancestor_synset)
        return synset in self.get_descendants(potential_ancestor_synset)

    def is_ancestor(self, synset, potential_descendant_synset):
        """
        Check whether a given class is an ancestor of another class.

        :param synset: The class name being searched for as an ancestor.
        :param potential_descendant_synset: The class name being searched for as a descendant.
        :return: bool indicating whether synset is an ancestor of potential_descendant_synset.
        """
        assert self.is_valid_class(synset)
        assert self.is_valid_class(potential_descendant_synset)
        return synset in self.get_ancestors(potential_descendant_synset)

    def get_abilities(self, synset):
        """
        Get the abilities of a given class.

        :param synset: Class name to search.
        :return: dict in the form of {ability: {param: value}} containing abilities and ability parameters.
        """
        assert self.is_valid_class(synset), f"Invalid class name: {synset}"
        return copy.deepcopy(self.taxonomy.nodes[synset]['abilities'])

    def get_categories(self, synset):
        """
        Get the iGibson object categories matching a given class.

        :param synset: Class name to search.
        :return: list of str corresponding to iGibson object categories matching the class.
        """
        assert self.is_valid_class(synset)
        return list(self.taxonomy.nodes[synset]['categories'])

    def get_children(self, synset):
        """
        Get the immediate child classes of a class.

        :param synset: Class name to search.
        :return: list of str corresponding to child class names.
        """
        assert self.is_valid_class(synset)
        return list(self.taxonomy.successors(synset))

    def get_parent(self, synset):
        """
        Get the immediate parent class of a class.

        :param synset: Class name to search.
        :return: str corresponding to parent class name, None if no parent exists.
        """
        assert self.is_valid_class(synset)

        in_degree = self.taxonomy.in_degree(synset)
        assert in_degree <= 1

        return next(self.taxonomy.predecessors(synset)) if in_degree else None

    def is_leaf(self, synset):
        """
        Check whether a given class is a leaf class e.g. it has no descendants.

        :param synset: Class name to search.
        :return: bool indicating if the class is a leaf class.
        """
        assert self.is_valid_class(synset), "{} is not a valid class name".format(synset)
        return self.taxonomy.out_degree(synset) == 0

    def has_ability(self, synset, ability):
        """
        Check whether the given class has the given ability.

        :param synset: Class name to check.
        :param ability: Ability name to check.
        :return: bool indicating if the class has the ability.
        """
        return ability in self.get_abilities(synset)


if __name__ == "__main__":
    object_taxonomy = ObjectTaxonomy()
    embed()

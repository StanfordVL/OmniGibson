'''
This code is lightly adapted from https://github.com/pucrs-automated-planning/pddl-parser
'''

import itertools
import re
import sys

from tasknet.config import SUPPORTED_PDDL_REQUIREMENTS as supported_requirements
from tasknet.config import get_definition_filename


def scan_tokens(filename):
    with open(filename, 'r') as f:
        # Remove single line comments
        str = re.sub(r';.*$', '', f.read(), flags=re.MULTILINE).lower()
    # Tokenize
    stack = []
    tokens = []
    for t in re.findall(r'[()]|[^\s()]+', str):
        if t == '(':
            stack.append(tokens)
            tokens = []
        elif t == ')':
            if stack:
                toks = tokens
                tokens = stack.pop()
                tokens.append(toks)
            else:
                raise Exception('Missing open parenthesis')
        else:
            tokens.append(t)
    if stack:
        raise Exception('Missing close parenthesis')
    if len(tokens) != 1:
        raise Exception('Malformed expression')
    return tokens[0]


def parse_domain(atus_activity, instance):
    domain_filename = get_definition_filename(atus_activity, instance, domain=True)
    tokens = scan_tokens(domain_filename)
    if type(tokens) is list and tokens.pop(0) == 'define':
        domain_name = 'unknown'
        requirements = []
        types = []
        actions = []
        predicates = {}
        while tokens:
            group = tokens.pop(0)
            t = group.pop(0)
            if t == 'domain':
                domain_name = group[0]
            elif t == ':requirements':
                for req in group:
                    if not req in supported_requirements:
                        raise Exception('Requirement %s not supported' % req)
                requirements = group 
            elif t == ':predicates':
                predicate_name, arguments = parse_predicates(group)
                if predicate_name in predicates:
                    raise Exception('Predicate %s defined multiple times' % predicate_name)
                predicates[predicate_name] = arguments
            elif t == ':types':
                types = group
            elif t == ':action':
                name = group.pop(0)
                for act in actions:
                    if act.name == name:
                        raise Exception('Action %s is defined multiple times' % name)
                actions.append(parse_action(group))
            else: 
                print('%s is not recognized in domain' % t)
        return domain_name, requirements, types, actions, predicates
    else:
        raise Exception('File %s does not match domain pattern' % domain_filename)


def parse_predicates(group):
    for pred in group:
        predicate_name = pred.pop(0)
        arguments = {}
        untyped_variables = []
        while pred:
            t = pred.pop(0)
            if t == '-':
                if not untyped_variables:
                    raise Exception('Unexpected hyphen in predicates')
                var_type = pred.pop(0)
                while untyped_variables:
                    arguments[untyped_variables.pop(0)] = var_type
            else:
                untyped_variables.append(t)
        while untyped_variables:
            arguments[untyped_variables.pop(0)] = 'object'
        return predicate_name, arguments


def parse_action(group):
    name = group.pop(0)
    if not isinstance(name, str):
        raise Exception('Action without name definition')
    parameters = []
    positive_preconditions = []
    negative_preconditions = []
    add_effects = []
    del_effects = []
    while group:
        t = group.pop(0)
        if t == ':parameters':
            if not isinstance(group, list):
                raise Exception('Error with %s parameters' % name)
            parameters = []
            untyped_parameters = []
            p = group.pop(0)
            while p:
                t = p.pop(0)
                if t == '-':
                    if not untyped_parameters:
                        raise Exception('Unexpected hyphen in %s parameters' % name)
                    param_type = p.pop(0)
                    while untyped_parameters:
                        parameters.append([untyped_parameters.pop(0), param_type])
                else:
                    untyped_parameters.append(t)
            while untyped_parameters:
                parameters.append([untyped_parameters.pop(0), 'object'])
        elif t == ':precondition':
            split_predicates(group.pop(0), positive_preconditions, negative_preconditions, name, ' preconditions')
        elif t == ':effect':
            split_predicates(group.pop(0), add_effects, del_effects, name, ' effects')
        else: 
            print('%s is not recognized in action' % t)
    return Action(name, parameters, positive_preconditions, negative_preconditions, add_effects, del_effects)


def parse_problem(atus_activity, task_instance, domain_name):
    problem_filename = get_definition_filename(atus_activity, task_instance)
    tokens = scan_tokens(problem_filename)
    if isinstance(tokens, list) and tokens.pop(0) == 'define':
        problem_name = 'unknown'
        objects = {}
        initial_state = []
        goal_state = []
        while tokens:
            group = tokens.pop()
            t = group[0]
            if t == 'problem':
                problem_name = group[-1]
            elif t == ':domain':
                if domain_name != group[-1]:
                    raise Exception('Different domain specified in problem file')
            elif t == ':requirements':
                pass 
            elif t == ':objects':
                group.pop(0)
                object_list = []
                while group:
                    if group[0] == '-':
                        group.pop(0)
                        objects[group.pop(0)] = object_list
                        object_list = []
                    else:
                        object_list.append(group.pop(0))
                if object_list:
                    if not 'object' in objects:
                        objects['object'] = []
                    objects['object'] += object_list
            elif t == ':init':
                group.pop(0)
                initial_state = group
            elif t == ':goal':
                package_predicates(group[1], goal_state, '', 'goals')
            else:
                print('%s is not recognized in problem' % t)
        return problem_name, objects, initial_state, goal_state
    else:
        raise Exception('File %s does not match problem pattern' % problem_filename)
             

def split_predicates(group, pos, neg, name, part):
    if not isinstance(group, list):
        raise Exception('Error with ' + name + part)
    if group[0] == 'and':
        group.pop(0)
    else:
        group = [group]
    for predicate in group:
        if predicate[0] == 'not':
            if len(predicate) != 2:
                raise Exception('Unexpected not in ' + name + part)
            neg.append(predicate[-1])     # NOTE removed this because I want the negative goals to have "not"
        else:
            pos.append(predicate)


def package_predicates(group, goals, name, part):
    if not isinstance(group, list):
        raise Exception('Error with ' + name + part)
    if group[0] == 'and':
        group.pop(0)
    else:
        group = [group]
    for predicate in group:
        goals.append(predicate)


class Action(object):

    def __init__(self, name, parameters, positive_preconditions, negative_preconditions, add_effects, del_effects):
        self.name = name
        self.parameters = parameters
        self.positive_preconditions = positive_preconditions
        self.negative_preconditions = negative_preconditions
        self.add_effects = add_effects
        self.del_effects = del_effects

    def __str__(self):
        return 'action: ' + self.name + \
        '\n  parameters: ' + str(self.parameters) + \
        '\n  positive_preconditions: ' + str(self.positive_preconditions) + \
        '\n  negative_preconditions: ' + str(self.negative_preconditions) + \
        '\n  add_effects: ' + str(self.add_effects) + \
        '\n  del_effects: ' + str(self.del_effects) + '\n'

    def __eq__(self, other): 
        return self.__dict__ == other.__dict__

    def groundify(self, objects):
        if not self.parameters:
            yield self
            return
        type_map = []
        variables = []
        for var, type in self.parameters:
            type_map.append(objects[type])
            variables.append(var)
        for assignment in itertools.product(*type_map):
            positive_preconditions = self.replace(self.positive_preconditions, variables, assignment)
            negative_preconditions = self.replace(self.negative_preconditions, variables, assignment)
            add_effects = self.replace(self.add_effects, variables, assignment)
            del_effects = self.replace(self.del_effects, variables, assignment)
            yield Action(self.name, assignment, positive_preconditions, negative_preconditions, add_effects, del_effects)

    def replace(self, group, variables, assignment):
        g = []
        for pred in group:
            pred = list(pred)
            iv = 0
            for v in variables:
                while v in pred:
                    pred[pred.index(v)] = assignment[iv]
                iv += 1
            g.append(pred)
        return g


######### UTIL ##########

def add_pddl_whitespace(pddl_file="task_conditions/parsing_tests/test_app_output.pddl", string=None, save=True):
    if pddl_file is not None:
        with open(pddl_file, 'r') as f:
            raw_pddl = f.read()
    elif string is not None:
        raw_pddl = string

    total_characters = len(raw_pddl)

    nest_level = 0
    refined_pddl = ""
    new_block = ""
    char_i = 0
    last_paren_type = None
    while char_i < total_characters:
        if raw_pddl[char_i] == "(":
            new_block = '\n' + '    ' * nest_level + raw_pddl[char_i]  
            last_paren_type = "("
            char_i += 1
            while (raw_pddl[char_i] not in [' ', ')']) and char_i < total_characters:
                new_block += raw_pddl[char_i] 
                char_i += 1
            refined_pddl += new_block + raw_pddl[char_i]
            if raw_pddl[char_i] == ' ':
                nest_level += 1
        elif raw_pddl[char_i] == ")":
            nest_level -= 1 
            if last_paren_type == ")":
                refined_pddl += "\n" + '    ' * nest_level
            refined_pddl += raw_pddl[char_i] 
            last_paren_type = ")"
        else:
            refined_pddl += raw_pddl[char_i] 
        char_i += 1

    if save:
        with open('task_conditions/parsing_tests/test_app_output_whitespace.pddl', 'w') as f:
            f.write(refined_pddl)        

    return refined_pddl


def remove_pddl_whitespace(pddl_file='task_conditions/parsing_tests/test_app_output_whitespace.pddl', string=None, save=True):
    if pddl_file is not None:
        with open(pddl_file, 'r') as f:
            raw_pddl = f.read()
    elif string is not None:
        raw_pddl = string
    else:
        raise ValueError('No PDDL given.')

    pddl = ' '.join([substr.lstrip(' ') for substr in raw_pddl.split('\n')])
    print(pddl)
    pddl = [' ' + substr if substr[0] != ')' else substr for substr in pddl.split(' ') if substr]
    print()
    print(pddl)
    pddl = ''.join(pddl)[1:]

    with open('task_conditions/parsing_tests/test_app_output_nowhitespace.pddl', 'w') as f:
        f.write(pddl)
    
    return pddl


if __name__ == '__main__':
    if sys.argv[1] == 'add':
        refined_pddl = add_pddl_whitespace()
    if sys.argv[1] == 'remove':
        refined_pddl = remove_pddl_whitespace()
    # print(refined_pddl)
    # import sys, pprint 
    # atus_activity = sys.argv[1]
    # task_instance = sys.argv[2]
    # print('----------------------------')
    # # pprint.pprint(scan_tokens(atus_activity, instance))
    # print('----------------------------')
    # # pprint.pprint(scan_tokens(atus_activity, instance))
    # print('----------------------------')
    # domain_name, requirements, types, actions, predicates = parse_domain(atus_activity, task_instance)
    # problem_name, objects, initial_state, goal_state = parse_problem(atus_activity, task_instance, domain_name)
    # print('----------------------------')
    # print('Problem name:', problem_name)
    # print('Objects:', objects)
    # print('Initial state:', initial_state)
    # print('Goal state:', goal_state)

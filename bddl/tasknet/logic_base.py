from abc import abstractmethod, ABCMeta
import re

from future.utils import with_metaclass


class Sentence(with_metaclass(ABCMeta)):
    def __init__(self, scope, task, body, object_map):
        self.children = []
        self.child_values = []
        self.task = task
        self.body = body
        self.scope = scope
        self.object_map = object_map

    @abstractmethod
    def evaluate(self):
        pass


class AtomicPredicate(Sentence):
    def __init__(self, scope, task, body, object_map):
        super().__init__(scope, task, body, object_map)


class BinaryAtomicPredicate(AtomicPredicate):
    STATE_NAME = None

    def __init__(self, scope, task, body, object_map):
        super().__init__(scope, task, body, object_map)
        assert len(body) == 2, 'Param list should have 2 args'
        self.input1, self.input2 = [inp.strip('?') for inp in body]
        self.scope = scope

        self.get_ground_options()

    @abstractmethod
    def _evaluate(self, obj1, obj2):
        pass

    def evaluate(self):
        if (self.scope[self.input1] is not None) and (self.scope[self.input2] is not None):
            return self._evaluate(self.scope[self.input1], self.scope[self.input2])
        else:
            print('%s and/or %s are not mapped to simulator objects in scope' %
                  (self.input1, self.input2))

    @abstractmethod
    def _sample(self, obj1, obj2, binary_state):
        pass

    def sample(self, binary_state):
        if (self.scope[self.input1] is not None) and (self.scope[self.input2] is not None):
            return self._sample(self.scope[self.input1], self.scope[self.input2], binary_state)
        else:
            print('%s and/or %s are not mapped to simulator objects in scope' %
                  (self.input1, self.input2))

    def get_ground_options(self):
        new_input_terms = []
        for input_term in [self.input1, self.input2]:
            if re.search(r"\.n\.\d+_", input_term) is not None:
                new_input_term = input_term
            else:
                # If the string token is an object category, then there will
                # exist another object instance that also points to the same
                # simulator object. Use that object instance instead.
                sim_obj = self.scope[input_term]
                for dsl_term, other_sim_obj in self.scope.items():
                    if dsl_term != input_term and sim_obj == other_sim_obj:
                        new_input_term = dsl_term
            new_input_terms.append(new_input_term)

        self.flattened_condition_options = [[[self.STATE_NAME,
                                              new_input_terms[0],
                                              new_input_terms[1]]]]


class UnaryAtomicPredicate(AtomicPredicate):
    STATE_NAME = None

    def __init__(self, scope, task, body, object_map):
        super().__init__(scope, task, body, object_map)
        assert len(body) == 1, 'Param list should have 1 arg'
        self.input = body[0].strip('?')
        self.scope = scope

        self.flattened_condition_options = [[[self.STATE_NAME, self.input]]]

    @abstractmethod
    def _evaluate(self, obj):
        pass

    def evaluate(self):
        if self.scope[self.input] is not None:
            return self._evaluate(self.scope[self.input])
        else:
            print('%s is not mapped to a simulator object in scope' % self.input)
            return False

    @abstractmethod
    def _sample(self, obj, binary_state):
        pass

    def sample(self, binary_state):
        if self.scope[self.input] is not None:
            return self._sample(self.scope[self.input], binary_state)
        else:
            print('%s is not mapped to a simulator object in scope' % self.input)
            return False

    def get_ground_options(self):
        if '_' in self.input:
            input_term = self.input
        else:
            # If the string token is an object category, then there will
            # exist another object instance that also points to the same
            # simulator object. Use that object instance instead.
            sim_obj = self.scope[self.input]
            for dsl_term, other_sim_obj in self.scope.items():
                if dsl_term != input_term and sim_obj == other_sim_obj:
                    input_term = dsl_term
        self.flattened_condition_options = [[[self.STATE_NAME, input_term]]]

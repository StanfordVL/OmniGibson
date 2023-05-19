from bddl.logic_base import UnaryAtomicFormula, BinaryAtomicFormula
from bddl.backend_abc import BDDLBackend

unaries = ["cooked", "real", "future", "frozen", "closed", "open", "folded", "unfolded", "toggled_on", "hot", "on_fire", "assembled", "broken"]
binaries = ["saturated", "covered", "filled", "contains", "ontop", "nextto", "empty", "under", "touching", "inside", "overlaid", "attached", "draped", "insource", "inroom"]

class DebugUnaryFormula(UnaryAtomicFormula):
    def _evaluate():
        return True 
    def _sample():
        return True
    

class DebugBinaryFormula(BinaryAtomicFormula):
    def _evaluate():
        return True 
    def _sample():
        return True


def gen_unary_token(predicate_name, generate_ground_options=True):
    return type(f"{predicate_name}StateUnaryPredicate", (DebugUnaryFormula,), {"STATE_CLASS": "HowDoesItMatter", "STATE_NAME": predicate_name})


def gen_binary_token(predicate_name, generate_ground_options=True):
    return type(f"{predicate_name}StateBinaryPredicate", (DebugBinaryFormula,), {"STATE_CLASS": "HowDoesItMatter", "STATE_NAME": predicate_name})


class DebugBackend(BDDLBackend):
    def get_predicate_class(self, predicate_name):
        if predicate_name in unaries: 
            return gen_unary_token(predicate_name)
        elif predicate_name in binaries:
            return gen_binary_token(predicate_name)
        else: 
            raise KeyError(predicate_name)


class DebugGenericObject(object): 
    def __init__(self, name):
        self.name = name
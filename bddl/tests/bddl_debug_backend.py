from bddl.logic_base import UnaryAtomicFormula, BinaryAtomicFormula
from bddl.backend_abc import BDDLBackend
from bddl.parsing import parse_domain


*__, domain_predicates = parse_domain("omnigibson")
UNARIES = [predicate for predicate, inputs in domain_predicates.items() if len(inputs) == 1]
BINARIES = [predicate for predicate, inputs in domain_predicates.items() if len(inputs) == 2]


class DebugBackend(BDDLBackend):
    def get_predicate_class(self, predicate_name):
        PREDICATE_MAPPING = {
            "cooked": DebugCookedPredicate,
            "frozen": DebugFrozenPredicate,
            "covered": DebugCoveredPredicate,
            "ontop": DebugOntopPredicate,
            "inside": DebugInsidePredicate,
            "filled": DebugFilledPredicate
        } 
        return PREDICATE_MAPPING[predicate_name]


class DebugSimulator(object):
    def __init__(self):
        # Unaries - populated with 1-tuples of string names 
        self.cooked = set()
        self.frozen = set()
        self.open = set()
        self.folded = set()
        self.unfolded = set()
        self.toggled_on = set() 
        self.hot = set() 
        self.on_fire = set() 
        self.future = set() 
        self.real = set() 
        self.broken = set()
        self.assembled = set()
        # Binaries - populated with 2-tuples of string names
        self.saturated = set()
        self.covered = set() 
        self.filled = set() 
        self.contains = set() 
        self.ontop = set() 
        self.nextto = set() 
        self.under = set() 
        self.touching = set() 
        self.inside = set() 
        self.overlaid = set() 
        self.attached = set() 
        self.draped = set() 
        self.insource = set() 

        self.create_predicate_to_setters()
    
    def create_predicate_to_setters(self):
        self.predicate_to_setters = {
            "cooked": self.set_cooked,
            "frozen": self.set_frozen,
            "inside": self.set_inside,
            "ontop": self.set_ontop,
            "covered": self.set_covered,
            "filled": self.set_filled,
        }

    def set_state(self, literals): 
        """
        Given a set of non-contradictory parsed ground literals, set this backend to them. 
        """
        for literal in literals: 
            is_predicate = not(literal[0] == "not")
            predicate, *objects = literal[1] if (literal[0] == "not") else literal
            if predicate == "inroom": 
                print(f"Skipping inroom literal {literal}")
                continue
            self.predicate_to_setters[predicate](tuple(objects), is_predicate)

    def set_cooked(self, objs, is_cooked):
        assert len(objs) == 1
        if is_cooked: 
            self.cooked.add(obj.name)
        else: 
            self.cooked.discard(obj.name)
    
    def get_cooked(self, objs):
        return tuple(obj.name for obj in objs) in self.cooked
    
    def set_frozen(self, objs, is_frozen):
        assert len(objs) == 1
        if is_frozen: 
            self.frozen.add(obj)
        else: 
            self.frozen.discard(obj)
    
    def get_frozen(self, objs):
        return tuple(obj.name for obj in objs) in self.frozen
    
    # TODO remaining unaries 

    def set_covered(self, objs, is_covered):
        assert len(objs) == 2
        if is_covered:
            self.covered.add(objs)
        else:
            self.covered.discard(objs)

    def get_covered(self, objs):
        return tuple(obj.name for obj in objs) in self.covered
    
    def set_ontop(self, objs, is_ontop):
        assert len(objs) == 2
        if is_ontop:
            self.ontop.add(objs)
        else:
            self.ontop.discard(objs)

    def get_ontop(self, objs):
        return tuple(obj.name for obj in objs) in self.ontop
    
    def set_inside(self, objs, is_inside):
        assert len(objs) == 2
        if is_inside:
            self.inside.add(objs)
        else:
            self.inside.discard(objs)
    
    def get_inside(self, objs):
        return tuple(obj.name for obj in objs) in self.inside
    
    def set_filled(self, objs, is_filled):
        assert len(objs) == 2
        if is_filled:
            self.filled.add(objs)
        else:
            self.filled.discard(objs)
    
    def get_filled(self, objs):
        return tuple(obj.name for obj in objs) in self.filled

    
    # TODO remaining binaries


class DebugGenericObject(object): 
    def __init__(self, name, simulator):
        self.name = name
        self.simulator = simulator
    
    def get_cooked(self):
        return self.simulator.get_cooked((self,))
    
    def get_frozen(self):
        return self.simulator.get_frozen((self,))
    
    def get_ontop(self, other):
        return self.simulator.get_ontop((self, other))
    
    def get_covered(self, other):
        return self.simulator.get_covered((self, other))

    def get_inside(self, other):
        return self.simulator.get_inside((self, other))
    
    def get_filled(self, other):
        return self.simulator.get_filled((self, other))


# OmniGibson debug predicates
class DebugCookedPredicate(UnaryAtomicFormula):
    STATE_NAME = "cooked"

    def _evaluate(self, obj):
        print(self.STATE_NAME, obj.name, obj.get_cooked())
        return obj.get_cooked()

    def _sample(self, obj1, binary_state):
        pass


class DebugFrozenPredicate(UnaryAtomicFormula):
    STATE_NAME = "frozen"

    def _evaluate(self, obj):
        print(self.STATE_NAME, obj.name, obj.get_frozen())
        return obj.get_frozen()

    def _sample(self, obj1, binary_state):
        pass


class DebugCoveredPredicate(BinaryAtomicFormula):
    STATE_NAME = "covered"

    def _evaluate(self, obj1, obj2):
        print(self.STATE_NAME, obj1.name, obj2.name, obj1.get_covered(obj2))
        return obj1.get_covered(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


class DebugInsidePredicate(BinaryAtomicFormula):
    STATE_NAME = "inside"

    def _evaluate(self, obj1, obj2):
        print(self.STATE_NAME, obj1.name, obj2.name, obj1.get_inside(obj2))
        return obj1.get_inside(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


class DebugOntopPredicate(BinaryAtomicFormula):
    STATE_NAME = "ontop"

    def _evaluate(self, obj1, obj2):
        print(self.STATE_NAME, obj1.name, obj2.name, obj1.get_ontop(obj2))
        return obj1.get_ontop(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


class DebugFilledPredicate(BinaryAtomicFormula):
    STATE_NAME = "ontop"

    def _evaluate(self, obj1, obj2):
        print(self.STATE_NAME, obj1.name, obj2.name, obj1.get_ontop(obj2))
        return obj1.get_ontop(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


# TODO remaining debug predicates

# TODO sample functions where we do original setting


VALID_ATTACHMENTS = set([
    ("mixing_bowl.n.01", "electric_mixer.n.01"),
    ("cork.n.04", "wine_bottle.n.01"),
    ("menu.n.01", "wall.n.01"),
    ("broken__light_bulb.n.01", "table_lamp.n.01"),
    ("light_bulb.n.01", "table_lamp.n.01"),
    ("lens.n.01", "digital_camera.n.01"),
    ("screen.n.01", "wall.n.01"),
    ("antler.n.01", "wall.n.01"),
    ("skateboard_wheel.n.01", "skateboard.n.01"),
    ("blackberry.n.01", "scrub.n.01"),
    ("raspberry.n.02", "scrub.n.01"),
    ("dip.n.07", "candlestick.n.01"),
    ("sign.n.02", "wall.n.01"),
    ("wreath.n.01", "wall.n.01"),
    ("bow.n.08", "wall.n.01"),
    ("holly.n.03", "wall.n.01"),
    ("curtain_rod.n.01", "wall.n.01"),
    ("bicycle.n.01", "bicycle_rack.n.01"),
    ("bicycle_rack.n.01", "wall.n.01"),
    ("dartboard.n.01", "wall.n.01"),
    ("rug.n.01", "wall.n.01"),
    ("fairy_light.n.01", "wall.n.01"),
    ("lantern.n.01", "wall.n.01"),
    ("address.n.05", "wall.n.01"),
    ("hanger.n.02", "wardrobe.n.01"),
    ("flagpole.n.02", "wall.n.01"),
    ("picture_frame.n.01", "wall.n.01"),
    ("wind_chime.n.01", "pole.n.01"),
    ("pole.n.01", "wall.n.01"),
    ("hook.n.05", "trailer_truck.n.01"),
    ("fire_alarm.n.02", "wall.n.01"),
    ("poster.n.01", "wall.n.01"),
    ("painting.n.01", "wall.n.01"),
    ("hanger.n.02", "coatrack.n.01"),
    ("license_plate.n.01", "car.n.01"),
    ("gummed_label.n.01", "license_plate.n.01"),
    ("wallpaper.n.01", "wall.n.01"),
    ("mirror.n.01", "wall.n.01"),
    ("webcam.n.02", "desktop_computer.n.01"),
    ("kayak.n.01", "kayak_rack.n.01"),
    ("kayak_rack.n.01", "wall.n.01"),
    ("fish.n.02", "fishing_rod.n.01"),
    ("bicycle_rack.n.01", "recreational_vehicle.n.01"),
])

VALID_ROOMS = set()
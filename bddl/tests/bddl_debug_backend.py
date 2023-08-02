from bddl.logic_base import UnaryAtomicFormula, BinaryAtomicFormula
from bddl.backend_abc import BDDLBackend
from bddl.parsing import parse_domain


*__, domain_predicates = parse_domain("omnigibson")
UNARIES = [predicate for predicate, inputs in domain_predicates.items() if len(inputs) == 1]
BINARIES = [predicate for predicate, inputs in domain_predicates.items() if len(inputs) == 2]

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
        if predicate_name in UNARIES: 
            return gen_unary_token(predicate_name)
        elif predicate_name in BINARIES:
            return gen_binary_token(predicate_name)
        else: 
            raise KeyError(predicate_name)


class DebugGenericObject(object): 
    def __init__(self, name):
        self.name = name


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
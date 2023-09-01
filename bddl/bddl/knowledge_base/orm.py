from functools import cached_property
from collections import defaultdict
from dataclasses import field, fields
import uuid

class Model:
    _CLASS_REGISTRY = dict()
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Model._CLASS_REGISTRY[cls.__name__] = cls

    @staticmethod
    def get_model_class(target_model):
        return Model._CLASS_REGISTRY[target_model]

    _OBJECT_REGISTRY = defaultdict(dict)
    def __post_init__(self, *args, **kwargs):
        key = self._key
        assert key not in self._OBJECT_REGISTRY[self.__class__.__name__], f"Duplicate key {key} for {self.__class__.__name__}"
        self._OBJECT_REGISTRY[self.__class__.__name__][key] = self

        # Also bind the "this_instance" member of each fk field
        for field in fields(self):
            if field.type == ManyToOne:
                fk = getattr(self, field.name)
                fk._this_instance = self
            elif field.type == OneToMany:
                fk = getattr(self, field.name)
                fk._this_instance = self
            elif field.type == ManyToMany:
                fk = getattr(self, field.name)
                fk._this_instance = self

    @cached_property
    def _key(self):
        return tuple([getattr(self, self.Meta.pk)])

    @classmethod
    def get(cls, *args):
        return cls._OBJECT_REGISTRY[cls.__name__][args]
    
    @classmethod
    def all_objects(cls):
        return cls._OBJECT_REGISTRY[cls.__name__].values()
    
    def __lt__(self, other):
        if hasattr(self.Meta, "ordering"):
            order_attrs = self.Meta.ordering
        else:
            order_attrs = self._key
        self_order_key = [getattr(self, attr) for attr in order_attrs]
        other_order_key = [getattr(other, attr) for attr in order_attrs]
        return self_order_key < other_order_key
    
    def __getattr__(self, key):
        if hasattr(self, key + "_fk"):
            fk = getattr(self, key + "_fk")
            if isinstance(fk, ManyToOne):
                return fk.get()
            else:
                return fk
        else:
            super().__getattr__(key)

    def __setattr__(self, key, value):
        if hasattr(self, key + "_fk"):
            fk = getattr(self, key + "_fk")
            if isinstance(fk, ManyToOne):
                return fk.set(self, value)
            else:
                raise ValueError(f"Cannot set {key} because it is a OneToMany or ManyToMany field.")
        else:
            super().__setattr__(key, value)

def has_field_with_type(cls, field_name, field_type):
    cls_fields = [x for x in fields(cls) if x.name == field_name]
    if not cls_fields:
        return False
    field, = cls_fields
    return field.type == field_type

def get_model_class(target_model):
    if isinstance(target_model, str):
        return Model.get_model_class(target_model)
    elif isinstance(target_model, type) and issubclass(target_model, Model):
        return target_model
    else:
        raise ValueError(f"Invalid target model {target_model}")

class ManyToOne:
    def __init__(self, target_model, target_field):
        target_model = get_model_class(target_model)
        target_field = target_field + "_fk"
        assert has_field_with_type(target_model, target_field, OneToMany)

        self._target_model = target_model
        self._target_field = target_field

        self._target_key = None
        self._this_inst = None
    
    def get(self):
        assert self._this_inst is not None, "Unbound foreign key object"
        return self._target_model.get(self._target_key)
    
    def set(self, target_inst):
        assert self._this_inst is not None, "Unbound foreign key object"
        assert target_inst.__class__ == self._target_model, f"Target instance {target_inst} is not of type {self._target_model}"
        self._target_key = target_inst._key
        getattr(target_inst, self._target_field)._values.add(self._this_inst)

class OneToMany:
    def __init__(self, target_model, target_field) -> None:
        target_model = get_model_class(target_model)
        target_field = target_field + "_fk"
        assert has_field_with_type(target_model, target_field, ManyToOne)

        self._target_model = target_model
        self._target_field = target_field

        self._values = set()
        self._this_inst = None

    def __iter__(self):
        assert self._this_inst is not None, "Unbound foreign key object"
        return iter(self._values)

class ManyToMany:
    def __init__(self, target_model, target_field) -> None:
        target_model = get_model_class(target_model)
        target_field = target_field + "_fk"
        assert has_field_with_type(target_model, target_field, ManyToMany)

        self._target_model = target_model
        self._target_field = target_field

        self._values = set()
        self._this_inst = None

    def __iter__(self):
        assert self._this_inst is not None, "Unbound foreign key object"
        return iter(self._values)
    
    def add(self, this_inst, target_inst):
        assert self._this_inst is not None, "Unbound foreign key object"
        assert target_inst.__class__ == self._target_model, f"Target instance {target_inst} is not of type {self._target_model}"
        self._values.add(target_inst)
        getattr(target_inst, self._target_field)._values.add(self._this_inst)

def ManyToOneField(target_model, target_field, **kwargs):
    return field(default_factory=lambda: ManyToOne(target_model, target_field), **kwargs)

def OneToManyField(target_model, target_field, **kwargs):
    return field(default_factory=lambda: OneToMany(target_model, target_field), **kwargs)

def ManyToManyField(target_model, target_field, **kwargs):
    return field(default_factory=lambda: ManyToMany(target_model, target_field), **kwargs)

def UUIDField(**kwargs):
    return field(default_factory=lambda: uuid.uuid4(), **kwargs)
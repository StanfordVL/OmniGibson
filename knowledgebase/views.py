import re
from flask.views import View
from flask import render_template
from bddl.knowledge_base import *


def camel_to_snake(name: str) -> str:
    """Converts a CamelCase string to snake_case."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


class TemplateView(View):
    def get_context_data(self):
        return {"view": self}

    def get_template_name(self):
        return self.template_name

    def dispatch_request(self, **kwargs):
        return render_template(self.get_template_name(), **self.get_context_data(**kwargs))


class ListView(TemplateView):
    def get_template_name(self):
        try:
            return super().get_template_name()
        except AttributeError:
            return f"{camel_to_snake(self.model.__name__)}_list.html"

    def get_queryset(self):
        return list(self.model.all_objects())

    def get_context_data(self):
        context = super().get_context_data()
        context[self.context_object_name] = self.get_queryset()
        return context


class DetailView(TemplateView):
    model = Category
    context_object_name = "category"
    slug_field = "name"
    slug_url_kwarg = "category_name"

    def get_template_name(self):
        try:
            super().get_template_name()
        except AttributeError:
            return f"{camel_to_snake(self.model.__name__)}_detail.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data()
        assert len(kwargs) == 1 and self.slug_url_kwarg in kwargs
        lookup_kwargs = {self.slug_field: kwargs[self.slug_url_kwarg]}
        context[self.context_object_name] = self.model.get(**lookup_kwargs)
        return context


class TaskListView(ListView):
    model = Task
    context_object_name = "task_list"


class TransitionFailureTaskListView(TaskListView):
    page_title = "Transition Failure Tasks"

    def get_queryset(self):
        return [x for x in super().get_queryset() if not x.goal_is_reachable]
    

class NonSceneMatchedTaskListView(TaskListView):
    page_title = "Non-Scene-Matched Tasks"

    def get_queryset(self):
        return [x for x in super().get_queryset() if x.scene_state == STATE_UNMATCHED]


class ObjectListView(ListView):
    model = Object
    context_object_name = "object_list"


class SubstanceMappedObjectListView(ObjectListView):
    page_title = "Objects Incorrectly Mapped to Substance Synsets"

    def get_queryset(self):
        return [x for x in super().get_queryset() if x.category.synset.state == STATE_SUBSTANCE]


class SceneListView(ListView):
    model = Scene
    context_object_name = "scene_list"


class SynsetListView(ListView):
    model = Synset
    context_object_name = "synset_list"
    wide = True

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["properties"] = sorted({p.name for p in Property.all_objects()})
        return context


class CategoryListView(ListView):
    model = Category
    context_object_name = "category_list"


class NonLeafSynsetListView(SynsetListView):
    page_title = "Non-Leaf Object-Assigned Synsets"

    def get_queryset(self):
        return [
            s for s in super().get_queryset()
            if sum([len(c.objects) for c in s.categories]) > 0 and len(s.children) > 0
        ]
    

class SubstanceErrorSynsetListView(SynsetListView):
    page_title = "Synsets Used in Wrong (Substance/Rigid) Predicates"

    def get_queryset(self):
        return [
            s for s in super().get_queryset()
            if (
                (s.state == STATE_SUBSTANCE and s.is_used_as_non_substance) or 
                (not s.state == STATE_SUBSTANCE and s.is_used_as_substance) or 
                (s.is_used_as_substance and s.is_used_as_non_substance)
            )]
    

class FillableSynsetListView(SynsetListView):
    page_title = "Synsets Used as Fillables"

    def get_queryset(self):
        return [s for s in super().get_queryset() if s.is_used_as_fillable]


class UnsupportedPropertySynsetListView(SynsetListView):
    page_title = "Task-Relevant Synsets with Object-Unsupported Properties"

    def get_queryset(self):
        return [
            s for s in super().get_queryset()
            if not s.has_fully_supporting_object
        ]


class TransitionListView(ListView):
    model = TransitionRule
    context_object_name = "transition_list"


class TaskDetailView(DetailView):
    model = Task
    context_object_name = "task"
    slug_field = "name"
    slug_url_kwarg = "name"
    

class SynsetDetailView(DetailView):
    model = Synset
    context_object_name = "synset"
    slug_field = "name"
    slug_url_kwarg = "name"


class CategoryDetailView(DetailView):
    model = Category
    context_object_name = "category"
    slug_field = "name"
    slug_url_kwarg = "name"
        

class ObjectDetailView(DetailView):
    model = Object
    context_object_name = "object"
    slug_field = "name"
    slug_url_kwarg = "name" 
    

class SceneDetailView(DetailView):
    model = Scene
    context_object_name = "scene"
    slug_field = "name"
    slug_url_kwarg = "name"


class TransitionDetailView(DetailView):
    model = TransitionRule
    context_object_name = "transition"
    slug_field = "name"
    slug_url_kwarg = "name"


class IndexView(TemplateView):
    template_name = "index.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # task metadata
        tasks_state = [task.state for task in Task.all_objects()]
        context["task_metadata"] = [
            sum([1 for state in tasks_state if state == STATE_MATCHED]),
            sum([1 for state in tasks_state if state == STATE_PLANNED]),
            sum([1 for state in tasks_state if state == STATE_UNMATCHED]),
            sum([1 for x in Task.all_objects() if x.scene_state == STATE_UNMATCHED]),
            len(tasks_state)
        ]
        # synset metadata
        context["synset_metadata"] = [
            sum(1 for x in Synset.all_objects() if x.state == STATE_MATCHED),
            sum(1 for x in Synset.all_objects() if x.state == STATE_PLANNED),
            sum(1 for x in Synset.all_objects() if x.state == STATE_SUBSTANCE),
            sum(1 for x in Synset.all_objects() if x.state == STATE_UNMATCHED),
            sum(1 for x in Synset.all_objects() if x.state == STATE_ILLEGAL),
            sum(1 for x in Synset.all_objects()),
        ]
        # object metadata
        context["object_metadata"] = [
            sum(1 for x in Object.all_objects() if x.ready),
            sum(1 for x in Object.all_objects() if not x.ready and x.planned),
            sum(1 for x in Object.all_objects() if not x.planned),
        ]
        # scene metadata
        num_ready_scenes = sum([scene.any_ready for scene in Scene.all_objects()])
        num_planned_scenes = sum(1 for scene in Scene.all_objects()) - num_ready_scenes
        context["scene_metadata"] = [num_ready_scenes, num_planned_scenes]
        return context
    

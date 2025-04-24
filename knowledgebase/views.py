import inspect
import io
import re
from flask.views import View
from flask import render_template, send_file, url_for, jsonify
from bddl.knowledge_base import Task, Scene, Synset, Category, Object, TransitionRule, AttachmentPair, Property, SynsetState, ComplaintType

from . import profile_utils


def camel_to_snake(name: str) -> str:
    """Converts a CamelCase string to snake_case."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


class TemplateView(View):
    def get_context_data(self):
        return {"view": self, "SynsetState": SynsetState}

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
    page_title = inspect.getdoc(Task.view_transition_failure)

    def get_queryset(self):
        return Task.view_transition_failure()
    

class NonSceneMatchedTaskListView(TaskListView):
    page_title = inspect.getdoc(Task.view_non_scene_matched)

    def get_queryset(self):
        return Task.view_non_scene_matched()


class ChallengeTaskListView(TaskListView):
    page_title = inspect.getdoc(Task.view_challenge)

    def get_queryset(self):
        return Task.view_challenge()
    

class ObjectListView(ListView):
    model = Object
    context_object_name = "object_list"


class MissingMetaLinkObjectListView(ObjectListView):
    page_title = inspect.getdoc(Object.view_objects_with_missing_meta_links)

    def get_queryset(self):
        return Object.view_objects_with_missing_meta_links()


class SceneListView(ListView):
    model = Scene
    context_object_name = "scene_list"


class CategoryListView(ListView):
    model = Category
    context_object_name = "category_list"


class NonLeafCategoryListView(CategoryListView):
    page_title = inspect.getdoc(Category.view_mapped_to_non_leaf_synsets)

    def get_queryset(self):
        return Category.view_mapped_to_non_leaf_synsets()


class SynsetListView(ListView):
    model = Synset
    context_object_name = "synset_list"
    wide = True

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["properties"] = sorted({p.name for p in Property.all_objects()})
        return context


class SubstanceMismatchSynsetListView(SynsetListView):
    page_title = inspect.getdoc(Synset.view_substance_mismatch)

    def get_queryset(self):
        return Synset.view_substance_mismatch()
    

class UnsupportedPropertySynsetListView(SynsetListView):
    page_title = inspect.getdoc(Synset.view_object_unsupported_properties)

    def get_queryset(self):
        return Synset.view_object_unsupported_properties()


class UnnecessarySynsetListView(SynsetListView):
    page_title = inspect.getdoc(Synset.view_unnecessary)

    def get_queryset(self):
        return Synset.view_unnecessary()


class BadDerivativeSynsetView(SynsetListView):
    page_title = inspect.getdoc(Synset.view_bad_derivative)

    def get_queryset(self):
        return Synset.view_bad_derivative()


class MissingDerivativeSynsetView(SynsetListView):
    page_title = inspect.getdoc(Synset.view_missing_derivative)

    def get_queryset(self):
        return Synset.view_missing_derivative()


class TransitionListView(ListView):
    model = TransitionRule
    context_object_name = "transition_list"


class AttachmentPairListView(ListView):
    model = AttachmentPair
    context_object_name = "attachment_pair_list"


class MissingObjectAttachmentPairListView(AttachmentPairListView):
    page_title = inspect.getdoc(AttachmentPair.view_attachment_pairs_with_missing_objects)

    def get_queryset(self):
        return AttachmentPair.view_attachment_pairs_with_missing_objects()


class ComplaintTypeListView(ListView):
    page_title = "Unresolved QA Complaint Types"
    model = ComplaintType
    context_object_name = "complaint_type_list"


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


class ComplaintTypeDetailView(DetailView):
    model = ComplaintType
    context_object_name = "complaint_type"
    slug_field = "name"
    slug_url_kwarg = "name"


class AttachmentPairDetailView(DetailView):
    model = AttachmentPair
    context_object_name = "attachment_pair"
    slug_field = "name"
    slug_url_kwarg = "name"


class IndexView(TemplateView):
    template_name = "index.html"

    def __init__(self, error_url_patterns, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_url_patterns = error_url_patterns

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # task metadata
        context["task_metadata"] = {
            "ready": sum([1 for task in Task.all_objects() if task.synset_state == SynsetState.MATCHED and task.scene_state == SynsetState.MATCHED]),
            "missing_object": sum([1 for task in Task.all_objects() if task.synset_state == SynsetState.UNMATCHED]),
            "missing_scene": sum([1 for task in Task.all_objects() if task.scene_state == SynsetState.UNMATCHED]),
            "total": len(list(Task.all_objects())),
        }
        # synset metadata
        context["synset_metadata"] = {
            "valid": sum(1 for x in Synset.all_objects() if x.state == SynsetState.MATCHED),
            "planned": sum(1 for x in Synset.all_objects() if x.state == SynsetState.PLANNED),
            "substance": sum(1 for x in Synset.all_objects() if x.state == SynsetState.SUBSTANCE),
            "unmatched": sum(1 for x in Synset.all_objects() if x.state == SynsetState.UNMATCHED),
            "illegal": sum(1 for x in Synset.all_objects() if x.state == SynsetState.ILLEGAL),
            "total": sum(1 for x in Synset.all_objects()),
        }
        # object metadata
        context["object_metadata"] = {
            "ready": sum(1 for x in Object.all_objects() if x.state == SynsetState.MATCHED),
            "planned": sum(1 for x in Object.all_objects() if x.state == SynsetState.PLANNED),
            "error": sum(1 for x in Object.all_objects() if x.state == SynsetState.UNMATCHED),
        }
        # scene metadata
        num_ready_scenes = sum([scene.any_ready for scene in Scene.all_objects()])
        num_planned_scenes = sum(1 for scene in Scene.all_objects()) - num_ready_scenes
        context["scene_metadata"] = [num_ready_scenes, num_planned_scenes]
        context["error_views"] = [
            (view_name, view_class.page_title, len(view_class().get_queryset()))
            for url, view_class, view_name in self.error_url_patterns
        ]
        return context
    

def searchable_items_list():
    SEARCHABLE_ITEM_TYPES = [
        (Object, "Object", "object_detail"),
        (Category, "Category", "category_detail"), 
        (Synset, "Synset", "synset_detail"),
        (Task, "Task", "task_detail"),
        (Scene, "Scene", "scene_detail"),
        (TransitionRule, "Transition Rule", "transition_detail"),
        (AttachmentPair, "Attachment Pair", "attachment_pair_detail")
    ]
    searchable_items = []
    for item_type, type_str, detail_view_name in SEARCHABLE_ITEM_TYPES:
        for item in item_type.all_objects():
            item_title = item.name if item_type != Object else str(item)
            searchable_items.append({"type": type_str, "title": item_title, "url": url_for(detail_view_name, name=item.name)})
    return jsonify(searchable_items)


def profile_plot_view():
    plot_img = profile_utils.plot_profile("Realtime Performance", 10, ignore_series=["Empty scene"])
    stream = io.BytesIO()
    plot_img.save(stream, format="png")
    stream.seek(0)
    return send_file(stream, mimetype="image/png")


def profile_badge_view():
    badge_text = profile_utils.make_realtime_badge("Rs_int")
    stream = io.BytesIO()
    stream.write(badge_text.encode("utf-8"))
    stream.seek(0)
    return send_file(stream, mimetype='image/svg+xml')

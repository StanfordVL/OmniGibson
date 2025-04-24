import unicodedata, re
from flask import redirect
from . import create_app
from knowledgebase.views import *

error_url_patterns = [
  ("transitionfailuretasks/", TransitionFailureTaskListView, "transition_failure_task_list"),
  ("nonscenematchedtasks/", NonSceneMatchedTaskListView, "non_scene_matched_task_list"),
  ("missingmetalinkobjects/", MissingMetaLinkObjectListView, "missing_meta_link_object_list"),
  ("substancemismatchsynsets/", SubstanceMismatchSynsetListView, "substance_mismatch_synset_list"),
  ("unsupportedpropertysynsets/", UnsupportedPropertySynsetListView, "unsupported_property_synset_list"),
  ("unnecessarysynsets/", UnnecessarySynsetListView, "unnecessary_synset_list"),
  ("badderivativesynsets/", BadDerivativeSynsetView, "bad_derivative_synset_list"),
  ("missingderivativesynsets/", MissingDerivativeSynsetView, "missing_derivative_synset_list"),
  ("nonleafcategories/", NonLeafCategoryListView, "non_leaf_category_list"),
  ("missingobjectattachmentpairs/", MissingObjectAttachmentPairListView, "missing_object_attachment_pair_list"),
  ("complainttypes/", ComplaintTypeListView, "complaint_type_list"),
]

urlpatterns = [
  ("", IndexView, "index", dict(error_url_patterns=error_url_patterns)),
  ("tasks/", TaskListView, "task_list"),
  ("objects/", ObjectListView, "object_list"),
  ("scenes/", SceneListView, "scene_list"),
  ("synsets/", SynsetListView, "synset_list"),
  ("categories/", CategoryListView, "category_list"),
  ("transitions/", TransitionListView, "transition_list"),
  ("attachments/", AttachmentPairListView, "attachment_pair_list"),
  ("tasks/<name>/", TaskDetailView, "task_detail"),
  ("synsets/<name>/", SynsetDetailView, "synset_detail"),
  ("categories/<name>/", CategoryDetailView, "category_detail"),
  ("scenes/<name>/", SceneDetailView, "scene_detail"),
  ("objects/<name>/", ObjectDetailView, "object_detail"),
  ("transitions/<name>/", TransitionDetailView, "transition_detail"),
  ("attachments/<name>/", AttachmentPairDetailView, "attachment_pair_detail"),
  ("complainttypes/<name>/", ComplaintTypeDetailView, "complaint_type_detail"),
  ("challengetasks/", ChallengeTaskListView, "challenge_task_list"),
]

app = create_app()
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True  # Add this line to enable Jinja2 auto-reload
app.debug = True  # This enables auto-reload for Python files

@app.template_filter('slugify')
def slugify_filter(value):
  value = str(value)
  value = (
    unicodedata.normalize("NFKD", value)
    .encode("ascii", "ignore")
    .decode("ascii")
  )
  value = re.sub(r"[^\w\s-]", "", value.lower())
  return re.sub(r"[-\s]+", "-", value).strip("-_")

@app.route("/", methods=["GET"])
def redirect_index():
  return redirect("/knowledgebase", code=302)

for view_info in urlpatterns + error_url_patterns:
  urlpattern, view_class, view_arg = view_info[:3]
  view_kwargs = view_info[3] if len(view_info) > 3 else {}
  view = view_class.as_view(view_arg, **view_kwargs)
  app.add_url_rule("/knowledgebase/" + urlpattern, view_func=view)

# Add the profile views
app.add_url_rule("/knowledgebase/profile/badge.svg", view_func=profile_badge_view)
app.add_url_rule("/knowledgebase/profile/plot.png", view_func=profile_plot_view)
app.add_url_rule("/knowledgebase/searchable_items.json", view_func=searchable_items_list)

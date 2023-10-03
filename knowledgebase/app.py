import unicodedata, re
from flask import Flask, redirect
from knowledgebase.views import *

urlpatterns = [
  ("", IndexView.as_view("index")),
  ("tasks/", TaskListView.as_view("task_list")),
  ("nonscenematchedtasks/", NonSceneMatchedTaskListView.as_view("non-scene-matched_task_list")),
  ("transitionfailuretasks/", TransitionFailureTaskListView.as_view("transition_failure_task_list")),
  ("objects/", ObjectListView.as_view("object_list")),
  ("substancemappedobjects/", SubstanceMappedObjectListView.as_view("substance_mapped_object_list")),
  ("scenes/", SceneListView.as_view("scene_list")),
  ("synsets/", SynsetListView.as_view("synset_list")),
  ("categories/", CategoryListView.as_view("category_list")),
  ("nonleafsynsets/", NonLeafSynsetListView.as_view("non-leaf_synset_list")),
  ("substanceerrorsynsets/", SubstanceErrorSynsetListView.as_view("substance_error_synset_list")),
  ("fillablesynsets/", FillableSynsetListView.as_view("fillable_synset_list")),
  ("unsupportedpropertysynsets/", UnsupportedPropertySynsetListView.as_view("unsupported_property_synset_list")),
  ("transitions/", TransitionListView.as_view("transition_list")),
  ("tasks/<name>/", TaskDetailView.as_view("task_detail")),
  ("synsets/<name>/", SynsetDetailView.as_view("synset_detail")),
  ("categories/<name>/", CategoryDetailView.as_view("category_detail")),
  ("scenes/<name>/", SceneDetailView.as_view("scene_detail")),
  ("objects/<name>/", ObjectDetailView.as_view("object_detail")),
  ("transitions/<name>/", TransitionDetailView.as_view("transition_detail")),
]

app = Flask(__name__)

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

for urlpattern, view in urlpatterns:
  app.add_url_rule("/knowledgebase/" + urlpattern, view_func=view)
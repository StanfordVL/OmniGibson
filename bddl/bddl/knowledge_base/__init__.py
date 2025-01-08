from bddl.knowledge_base.models import Property, \
  MetaLink, \
  Predicate, \
  Scene, \
  Category, \
  Object, \
  Synset, \
  TransitionRule, \
  Task, \
  RoomRequirement, \
  RoomSynsetRequirement, \
  Room, \
  RoomObject

from bddl.knowledge_base.utils import State

from bddl.knowledge_base.processing import KnowledgeBaseProcessor

# Load the knowledge base
KnowledgeBaseProcessor(verbose=False).run()

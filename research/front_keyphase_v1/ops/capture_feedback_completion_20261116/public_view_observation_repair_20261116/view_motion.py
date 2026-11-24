from view_candidate import ViewLesson
from train_relation_views import ViewRelation
class ViewMotionLesson(ViewLesson):
 def __init__(self):super().__init__();self.body_relation=ViewRelation()

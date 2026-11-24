"""Isolated dependency injection for the research web's signed-direction observer.

Existing consumer/capture source and neural weights are read-only. Bind their
unchanged methods to copied global namespaces so only refine and the consumer
constructor differ. The original module dictionaries are never modified.
"""
import copy,functools,types
from consumer import FeedbackConsumer as OriginalConsumer
from capture import CaptureService as OriginalCapture
from coaching import compose
from localized_direction import refine,localize_cues

def bind_dependency(function,name,value):
    if name not in function.__globals__:raise RuntimeError('Frozen dependency contract changed: '+name)
    namespace=dict(function.__globals__);namespace[name]=value
    replacement=types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    replacement.__kwdefaults__=function.__kwdefaults__
    return functools.update_wrapper(replacement,function)

class FeedbackConsumer(OriginalConsumer):
    channels=bind_dependency(OriginalConsumer.channels,'refine',refine)

class CaptureService(OriginalCapture):
    prepare=bind_dependency(OriginalCapture.prepare,'FeedbackConsumer',FeedbackConsumer)

def attach(result):
    out=copy.deepcopy(result);out['coaching']=localize_cues(result,compose);return out

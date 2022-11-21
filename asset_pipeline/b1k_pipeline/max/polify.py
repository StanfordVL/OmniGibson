import collections

import pymxs

rt = pymxs.runtime

def main():
  objs = [x for x in rt.objects if rt.classOf(x) == rt.Editable_Mesh]
  instmap = collections.Counter(x.baseObject for x in objs)

  for obj in objs:
    if instmap[obj.baseObject] == 1:
      print(f"Polifying {obj.name}")
      rt.convertToPoly(obj)

main()
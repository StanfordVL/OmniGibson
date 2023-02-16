import pymxs
rt = pymxs.runtime

def spherify(obj):
    assert rt.classOf(obj) == rt.Point, "Selected objects should be point helpers."
    s = rt.Sphere()
    s.position = obj.position
    s.name = obj.name
    s.radius = 20
    s.layer = obj.layer
    s.parent = obj.parent
    rt.delete(obj)
    return s

def main():
    rt.select([spherify(x) for x in rt.selection])

if __name__ == "__main__":
    main()
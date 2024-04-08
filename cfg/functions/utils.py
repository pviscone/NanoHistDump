import awkward as ak


def add_collection(events,name):
    events[name] = ak.Array([{}]*len(events))


def get_name(arr):
    return arr.layout.__repr__().split("<parameter name='collection_name'>'")[1].split("'</parameter>")[0]


def builder(func):
    def wrapper(*args, **kwargs):
        builder = ak.ArrayBuilder()
        return func(builder, *args, **kwargs).snapshot()

    return wrapper

def cartesian(obj1, obj2):
    name1 = get_name(obj1)
    name2 = get_name(obj2)

    cart = ak.cartesian([obj1, obj2])
    cart = ak.zip({name1: cart["0"], name2: cart["1"]})
    argcart = ak.argcartesian([obj1, obj2])

    cart[name1, "idx"] = argcart["0"]
    cart[name2, "idx"] = argcart["1"]
    return cart, name1, name2


"""
#maybe maybe maybe
def collection(name, hists, delete=None):
    def decorator(function):
        def wrapper(sample):
            add_collection(sample.events, name)
            result = function(sample)
            sample.add_hists(hists)
            if delete:
                del sample.events[name]
            return result
        return wrapper
    return decorator
 """
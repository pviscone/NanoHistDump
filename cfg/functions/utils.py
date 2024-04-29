import awkward as ak


def add_collection(events, name):
    events[name] = ak.Array([{}] * len(events))


def get_name(arr):
    try:
        if "name" not in arr._layout.content.parameters:
            return arr._layout.content.__repr__().split("<parameter name='collection_name'>'")[1].split("'</parameter>")[0]
        else:
            return arr._layout.content.parameters["name"]
    except:
        return arr._layout.__repr__().split("<parameter name='collection_name'>'")[1].split("'</parameter>")[0]

def set_name(arr, name):
    arr._layout.content.parameters["name"] = name




def builders(n=1):
    def decorator(function):
        def wrapper(*args, **kwargs):
            builders = [ak.ArrayBuilder() for _ in range(n)]
            res_list = function(*builders, *args, **kwargs)
            return [res.snapshot() for res in res_list]

        return wrapper

    return decorator


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

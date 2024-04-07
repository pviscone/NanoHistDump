import awkward as ak


def add_collection(events,name):
    events[name] = ak.Array([{}]*len(events))


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